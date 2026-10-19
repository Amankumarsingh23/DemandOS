"""TFT mechanisms implemented from the published formulas, not a TransformerEncoder."""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from demandos.neural.sequences import STATIC, PAST, FUTURE


class GLU(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.value = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        return self.value(x)*torch.sigmoid(self.gate(x))


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, *, context_size=None, dropout=0.):
        super().__init__()
        output_size = output_size or hidden_size
        self.input = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size, dropout)
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        hidden = self.input(x)
        if self.context is not None:
            if context is None:
                raise ValueError("Required static GRN context missing")
            while context.ndim < hidden.ndim:
                context = context.unsqueeze(1)
            hidden = hidden+self.context(context)
        elif context is not None:
            raise ValueError("This GRN has no context projection")
        return self.norm(self.skip(x)+self.glu(self.hidden(torch.nn.functional.elu(hidden))))


class VariableSelection(nn.Module):
    """Embedded variables [...,V,D]; softmax always over V, never over time."""
    def __init__(self, variables, hidden_size, *, context=False, dropout=0.):
        super().__init__()
        self.variables, self.hidden_size = variables, hidden_size
        self.weights = GRN(variables*hidden_size, hidden_size, variables,
                           context_size=hidden_size if context else None, dropout=dropout)
        self.transforms = nn.ModuleList([GRN(hidden_size, hidden_size, dropout=dropout) for _ in range(variables)])

    def forward(self, embedded, context=None):
        if embedded.shape[-2:] != (self.variables, self.hidden_size):
            raise ValueError("VSN requires a distinct variable and embedding axis")
        weights = torch.softmax(self.weights(embedded.flatten(-2), context), dim=-1)
        transformed = torch.stack([layer(embedded[...,i,:]) for i,layer in enumerate(self.transforms)], dim=-2)
        return (transformed*weights.unsqueeze(-1)).sum(dim=-2), weights


class InterpretableAttention(nn.Module):
    """Separate Q/K per head, one shared V, mean head outputs, not concatenation."""
    def __init__(self, hidden_size, heads, dropout=0.):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("Head dimension must divide model dimension")
        self.heads, self.depth = heads, hidden_size//heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, self.depth, bias=False)
        self.output = nn.Linear(self.depth, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, allowed):
        batch, queries_count, width = queries.shape
        if allowed.shape != (batch, queries_count, keys.shape[1]) or allowed.dtype != torch.bool or not allowed.any(-1).all():
            raise ValueError("Boolean attention mask must permit at least one key per query")
        q = self.query(queries).view(batch, queries_count, self.heads, self.depth).transpose(1,2)
        k = self.key(keys).view(batch, keys.shape[1], self.heads, self.depth).transpose(1,2)
        v = self.value(keys)
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.depth)
        weights = torch.softmax(scores.masked_fill(~allowed[:,None], float("-inf")), dim=-1)
        # Keep probability diagnostics before dropout; normalization remains auditable.
        combined = (self.dropout(weights) @ v[:,None]).mean(dim=1)
        return self.output(combined), weights


def attention_mask(history_mask, horizon):
    length = history_mask.shape[1]
    valid = torch.cat([history_mask.bool(), torch.ones(len(history_mask), horizon, dtype=torch.bool, device=history_mask.device)], dim=1)
    query_positions = torch.arange(length, length+horizon, device=history_mask.device)
    key_positions = torch.arange(length+horizon, device=history_mask.device)
    return valid[:,None,:] & (key_positions[None,:] <= query_positions[:,None])[None]


class TemporalFusionTransformer(nn.Module):
    def __init__(self, processor, config):
        super().__init__()
        self.config = config
        size, drop = config.hidden_size, config.dropout
        self.embeddings = nn.ModuleList([nn.Embedding(len(processor.vocabulary[name])+1, config.embedding_dim, padding_idx=0) for name in STATIC])
        self.static_projections = nn.ModuleList([nn.Linear(config.embedding_dim,size) for _ in STATIC])
        self.past_projections = nn.ModuleList([nn.Linear(1,size) for _ in PAST])
        self.future_projections = nn.ModuleList([nn.Linear(1,size) for _ in FUTURE])
        self.static_vsn = VariableSelection(len(STATIC),size,dropout=drop)
        self.past_vsn = VariableSelection(len(PAST),size,context=True,dropout=drop)
        self.future_vsn = VariableSelection(len(FUTURE),size,context=True,dropout=drop)
        self.context_selection = GRN(size,size,dropout=drop)
        self.context_hidden = GRN(size,size,dropout=drop)
        self.context_cell = GRN(size,size,dropout=drop)
        self.context_enrichment = GRN(size,size,dropout=drop)
        self.encoder = nn.LSTM(size,size,batch_first=True,num_layers=1,dropout=0.)
        self.decoder = nn.LSTM(size,size,batch_first=True,num_layers=1,dropout=0.)
        self.local_gate, self.local_norm = GLU(size,size,drop), nn.LayerNorm(size)
        self.enrichment = GRN(size,size,context_size=size,dropout=drop)
        self.attention = InterpretableAttention(size,config.heads,drop)
        self.attention_gate, self.attention_norm = GLU(size,size,drop), nn.LayerNorm(size)
        self.positionwise = GRN(size,size,dropout=drop)
        self.final_gate, self.final_norm = GLU(size,size,drop), nn.LayerNorm(size)
        self.head = nn.Linear(size,1)

    def encode_history(self, selected, valid, state):
        """Skip leading padding in the LSTM, including the zero-history case."""
        if valid.all():
            return self.encoder(selected,state)
        lengths = valid.sum(1).long()
        steps = torch.arange(selected.shape[1],device=selected.device)[None,:]
        index = (steps+(selected.shape[1]-lengths[:,None])).clamp(max=selected.shape[1]-1)
        compact = selected.gather(1,index[:,:,None].expand_as(selected))
        packed = pack_padded_sequence(compact,lengths.clamp(min=1).cpu(),batch_first=True,enforce_sorted=False)
        packed_output, encoded = self.encoder(packed,state)
        output,_ = pad_packed_sequence(packed_output,batch_first=True,total_length=selected.shape[1])
        reverse = (steps-(selected.shape[1]-lengths[:,None])).clamp(min=0)
        output = output.gather(1,reverse[:,:,None].expand_as(output))*valid[:,:,None]
        encoded = tuple(torch.where((lengths>0)[None,:,None],value,initial) for value,initial in zip(encoded,state))
        return output,encoded

    def forward(self, past, future, ids, *, diagnostics=False):
        c = self.config
        if past.ndim != 3 or past.shape[1:] != (c.lookback,len(PAST)) or future.shape != (len(past),c.horizon,len(FUTURE)) or ids.shape != (len(past),len(STATIC)):
            raise ValueError("Invalid TFT past/future/static tensor shapes")
        if not torch.isfinite(past).all() or not torch.isfinite(future).all():
            raise ValueError("Nonfinite TFT inputs")
        observed = past[:,:,1]
        if not ((observed==0)|(observed==1)).all() or (observed[:,1:] < observed[:,:-1]).any():
            raise ValueError("History mask must represent contiguous observed suffix after left padding")
        valid = observed.bool()
        safe_past = past*observed[:,:,None]
        static_inputs = torch.stack([projection(embedding(ids[:,i])) for i,(embedding,projection) in enumerate(zip(self.embeddings,self.static_projections))],dim=1)
        static,static_weights = self.static_vsn(static_inputs)
        context = self.context_selection(static)
        historical = torch.stack([projection(safe_past[:,:,i:i+1]) for i,projection in enumerate(self.past_projections)],dim=-2)
        known = torch.stack([projection(future[:,:,i:i+1]) for i,projection in enumerate(self.future_projections)],dim=-2)
        selected_past,past_weights = self.past_vsn(historical,context)
        selected_past = selected_past*observed[:,:,None]
        selected_future,future_weights = self.future_vsn(known,context)
        initial = (self.context_hidden(static)[None],self.context_cell(static)[None])
        past_local,state = self.encode_history(selected_past,valid,initial)
        future_local,_ = self.decoder(selected_future,state)
        selected = torch.cat([selected_past,selected_future],dim=1)
        local = self.local_norm(selected+self.local_gate(torch.cat([past_local,future_local],dim=1)))
        enriched = self.enrichment(local,self.context_enrichment(static))
        allowed = attention_mask(valid,c.horizon)
        attended,weights = self.attention(enriched[:,c.lookback:],enriched,allowed)
        attended = self.attention_norm(enriched[:,c.lookback:]+self.attention_gate(attended))
        fused = self.final_norm(local[:,c.lookback:]+self.final_gate(self.positionwise(attended)))
        prediction = self.head(fused).squeeze(-1)
        if diagnostics:
            return {"prediction":prediction,"static_weights":static_weights,"past_weights":past_weights,
                    "future_weights":future_weights,"attention":weights,"allowed":allowed,
                    "encoder_state_h":state[0],"encoder_state_c":state[1]}
        return prediction
