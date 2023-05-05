from ultralytics import YOLO

def run_test():
    model = YOLO(f'./ai_model/best.pt')
    model.predict(source="0", show=True)

run_test()