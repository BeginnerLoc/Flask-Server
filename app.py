from ai_server import create_app, socketio
import threading

app = create_app()


socketio.run(app)

