#!/usr/bin/env python3
import unittest
from src.extractors.complexity_analyzer import estimate_complexity

class TestComplexityAnalyzer(unittest.TestCase):
    def test_constant_complexity(self):
        code = """
        void process() {
            int a = 1;
            int b = 2;
            int c = a + b;
        }
        """
        self.assertEqual(estimate_complexity(code), "1")

    def test_linear_complexity(self):
        code = """
        void process(int n) {
            for (int i = 0; i < n; i++) {
                printf("%d\\n", i);
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "n")

    def test_logarithmic_complexity_division(self):
        code = """
        void process(int n) {
            for (int i = n; i > 0; i /= 2) {
                printf("%d\\n", i);
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "log n")

    def test_logarithmic_complexity_bitshift(self):
        code = """
        void process(int n) {
            int i = n;
            while (i > 0) {
                printf("%d\\n", i);
                i >>= 1;
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "log n")

    def test_quadratic_complexity(self):
        code = """
        void process(int n) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%d %d\\n", i, j);
                }
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "n^2")

    def test_n_log_n_complexity(self):
        code = """
        void process(int n) {
            for (int i = 0; i < n; i++) {
                for (int j = n; j > 0; j /= 2) {
                    printf("%d %d\\n", i, j);
                }
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "n log n")

    def test_cubic_complexity(self):
        code = """
        void process(int n) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        printf("%d\\n", i + j + k);
                    }
                }
            }
        }
        """
        self.assertEqual(estimate_complexity(code), "n^3")

if __name__ == "__main__":
    unittest.main()
