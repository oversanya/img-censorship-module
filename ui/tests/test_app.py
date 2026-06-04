from __future__ import annotations

import unittest

from ui.app import app, static_dir


class AppFrontendTests(unittest.TestCase):
    def test_registers_frontend_routes(self) -> None:
        route_paths = {getattr(route, "path", None) for route in app.routes}

        self.assertIn("/", route_paths)
        self.assertIn("/static", route_paths)
        self.assertTrue((static_dir / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
