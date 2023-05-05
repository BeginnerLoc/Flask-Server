from .extensions import socketio

@socketio.on("connect")
def handle_connect():
    print(f"================ Client connected ==============")
    
# @socketio.on("sendStream")
# def handle_stream_recieved(frame):
#     print("================ Receing frame ================")
#     socketio.emit('stream', frame, include_self=False)
#     print("================ Sending frame ================")



