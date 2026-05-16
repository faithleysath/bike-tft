from functools import partial
from pathlib import Path
import http.server
import socket
import ssl


PORT = 8001
DIRECTORY = Path(__file__).resolve().parent
CERT = "/etc/letsencrypt/live/tower.isok.dev/fullchain.pem"
KEY = "/etc/letsencrypt/live/tower.isok.dev/privkey.pem"


class DualStackThreadingHTTPServer(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIRECTORY))
httpd = DualStackThreadingHTTPServer(("::", PORT), handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=CERT, keyfile=KEY)
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
print(f"Serving HTTPS on :: port {PORT} from {DIRECTORY}", flush=True)
httpd.serve_forever()
