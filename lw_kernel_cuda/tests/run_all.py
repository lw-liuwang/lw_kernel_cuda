"""Run all tests."""
import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "--tb=short", __file__]))