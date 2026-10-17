"""Smoke check for the installed project foundation."""

import unittest
from importlib.metadata import version

import demandos


class PackageInstallationTest(unittest.TestCase):
    def test_installed_version_matches_package(self):
        self.assertEqual(version("demandos"), demandos.__version__)


if __name__ == "__main__":
    unittest.main()
