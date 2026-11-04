"""Interactive Project Defense Dashboard Server for TMG-GAN Research Project.

Uses Python's standard library http.server with REST API endpoints:
- GET  /api/status      : System status and hardware details
- GET  /api/flows       : Preloaded curated test flows across 6 classes
- POST /api/classify    : Multi-model prediction (Original vs SMOTE vs TMG-GAN)
- POST /api/generate    : TMG-GAN Algorithm 2 live synthetic generator
- GET  /api/metrics     : Benchmark comparison (Table VI, Benign FPR, per-class)
- Serves static dashboard assets from dashboard/
"""

import json
import mimetypes
import os
import sys
from functools import partial
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DASHBOARD_DIR = Path(__file__).absolute().parent
ROOT = DASHBOARD_DIR.parent
sys.path.append(str(ROOT))

from src.inference import get_inference_engine, CLASSES

RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = ROOT / 'data' / 'plots'

PORT = 8050


class DashboardAPIHandler(SimpleHTTPRequestHandler):
    """Handles both REST API calls and static file delivery."""

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/status':
            self.handle_status()
        elif path == '/api/flows':
            self.handle_flows()
        elif path == '/api/random_flow':
            params = parse_qs(parsed.query)
            filter_cls = params.get('class', [None])[0]
            self.handle_random_flow(filter_cls)
        elif path == '/api/metrics':
            self.handle_metrics()
        elif path.startswith('/api/plots/'):
            self.handle_plot_image(path.replace('/api/plots/', ''))
        else:
            # Fall back to static file handler (dashboard/index.html, styles.css, app.js)
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            body = json.loads(post_data.decode('utf-8'))
        except Exception:
            body = {}

        if path == '/api/classify':
            self.handle_classify(body)
        elif path == '/api/generate':
            self.handle_generate(body)
        elif path in ('/api/synthetic_test', '/api/adversarial_test'):
            self.handle_adversarial_test(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_json(self, data: dict, status: int = 200):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_status(self):
        engine = get_inference_engine()
        self.send_json({
            'status': 'ready',
            'device': str(engine.device),
            'feature_count': engine.feature_count,
            'classes': CLASSES,
            'models_available': list(engine.classifiers.keys()),
            'generators_available': len(engine.generators)
        })

    def handle_flows(self):
        engine = get_inference_engine()
        flows = engine.get_preloaded_flows()
        self.send_json({'flows': flows, 'count': len(flows)})

    def handle_classify(self, body: dict):
        engine = get_inference_engine()
        features = body.get('features')
        is_scaled = body.get('is_scaled', False)
        
        if features is None:
            self.send_json({'error': 'Missing features parameter'}, status=400)
            return

        try:
            comparison = engine.compare_models(features, is_scaled=is_scaled)
            self.send_json(comparison)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)

    def handle_generate(self, body: dict):
        engine = get_inference_engine()
        target_class_idx = body.get('class_index', 5) # Default to Bot
        seed = body.get('seed', None)

        try:
            sample = engine.generate_sample(target_class_idx=target_class_idx, seed=seed)
            self.send_json(sample)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)

    def handle_random_flow(self, filter_class: str = None):
        engine = get_inference_engine()
        flow = engine.get_random_flow(filter_class=filter_class)
        if flow is None:
            self.send_json({'error': 'No flow found'}, status=404)
        else:
            self.send_json(flow)

    def handle_adversarial_test(self, body: dict):
        engine = get_inference_engine()
        target_class_idx = body.get('class_index', 5)
        noise_scale = body.get('noise_scale', 1.0)
        seed = body.get('seed', None)

        try:
            report = engine.adversarial_stress_test(
                target_class_idx=target_class_idx,
                noise_scale=noise_scale,
                seed=seed
            )
            self.send_json(report)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)


    def handle_metrics(self):
        summary_csv = RESULTS_DIR / 'summary.csv'
        paper_csv = RESULTS_DIR / 'paper_vs_ours.csv'
        per_class_csv = RESULTS_DIR / 'per_class.csv'

        def read_csv_as_records(path: Path):
            if not path.exists():
                return []
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)

        self.send_json({
            'summary': read_csv_as_records(summary_csv),
            'paper_vs_ours': read_csv_as_records(paper_csv),
            'per_class': read_csv_as_records(per_class_csv)
        })

    def handle_plot_image(self, filename: str):
        # Look in results/ then data/plots/
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            file_path = PLOTS_DIR / filename
        
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Image {filename} not found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or 'image/png'

        with open(file_path, 'rb') as f:
            data = f.read()

        self.send_response(200)
        self.send_header('Content-Type', mime_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ('127.0.0.1', PORT)
    handler = partial(DashboardAPIHandler, directory=str(DASHBOARD_DIR))
    httpd = HTTPServer(server_address, handler)
    print("=" * 70)
    print("TMG-GAN DEFENSE & RESEARCH DEMONSTRATION SERVER")
    print("=" * 70)
    print(f"Server running at: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == '__main__':
    run_server()
