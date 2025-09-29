"""
Mathematical toolkit for symbolic computations using SymPy.
Includes methods for algebraic manipulation, calculus, and linear algebra.
"""

import json
from collections.abc import Callable

import sympy as sp

from utu.config import ToolkitConfig
from utu.utils import get_logger
from utu.tools import AsyncBaseToolkit

logger = get_logger(__name__)


class MathToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None, default_variable: str = 'x') -> None:
        super().__init__(config)
        self.default_variable = default_variable
        logger.info(f"Default variable set to: {self.default_variable}")

    def _handle_exception(self, method_name: str, e: Exception) -> str:
        """Handle exceptions and return formatted error message."""
        error_msg = f"Error in {method_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    async def calculator(self, expression: str) -> str:
        r"""Evaluates a mathematical expression.

        Args:
            expression (str): The mathematical expression to evaluate,
                provided as a string.

        Returns:
            str: JSON string containing the result of the evaluation in the
                `"result"` field. If an error occurs, the JSON string will
                include an `"error"` field with the corresponding error
                message.
        """
        try:
            expr = sp.parsing.sympy_parser.parse_expr(expression, evaluate=True)
            num = sp.N(expr, 15)  # enough precision
            val = float(num)
            formatted = ("{:.6f}".format(val)).rstrip('0').rstrip('.')
            return json.dumps({"result": formatted}, ensure_ascii=False)
        except Exception as e:
            return self._handle_exception("calculator", e)

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "calculator": self.calculator,
        }
