from ultralytics import solutions
import cv2
import torch   #pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 ## gpu hızlandırması için 


def main():
    source_dir = r".\videos\testvideo.mp4"
    model_dir = r".\models\yolov8x.pt"
    output = r".\results\çıktı1.avi"
    torch.cuda.set_device(0)
    # device = torch.device('cpu')
    device = torch.device('cuda')
    cap = cv2.VideoCapture(source_dir)
    if not cap.isOpened():
        print("no source file")
    else:

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(output, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
        h = int(h/2)
        # print(w , h , fps) ##debug

        counter = solutions.ObjectCounter(
            model=model_dir,
            classes=[2,3,5,7],
            # region=	[(20, h+50), (w-20, h+50)],
            # region = [(0,0),(w,0),(0,h),(w,h)],
            # region = [(0,h),(0,0),(w,h),(w,0)],
            region=[(0, h-100), (w, h-100)],
            up_angle=270,    #neden bilmiyorum çalışmıyor
            down_angle=90,   #neden bilmiyorum çalışmıyor
            show_in = True,
            show_out = True,
            device= device,
            tracker="bytetracker.yaml",
            conf = 0.3
            )
        # print(counter.CFG) #debug
        while True:
            success, frame = cap.read()
            if not success:
                print("boş kare geçildi veya video bitti")
                video_writer.release()
                break
            result = counter.count(frame)
            in_count = counter.in_count
            out_count = counter.out_count
            label_text = f"Count: {in_count + out_count}"
            font = cv2.QT_FONT_NORMAL
            font_scale = 1
            color = (56, 62, 66) 
            thickness = 2
            position = (50, 50) 
            top_left_corner = (40, 15)
            bottom_right_corner = (top_left_corner[0] + 200, top_left_corner[1] + 50)
            cv2.rectangle(result, top_left_corner, bottom_right_corner, (255, 255, 255), -1)
            cv2.putText(result, label_text, position, font, font_scale, color, thickness)
            cv2.imshow("sayac",result)
            # video_writer.write(result)
            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    main()