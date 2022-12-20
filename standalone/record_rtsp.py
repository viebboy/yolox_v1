import argparse
import threading
import cv2
import time
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser("record from RTSP")
    parser.add_argument("--rtsp-path", required=True, type=str, help="rtsp path")
    parser.add_argument("--output-file", required=True, type=str, help="path to output")
    parser.add_argument("--countdown", default=10, type=int, help="number of seconds before recording start")

    return parser.parse_args()


def capture_from_rtsp(path, image_queue, event_queue, skip_frame):
    cap = cv2.VideoCapture(path)
    skip_flag = False
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            if skip_frame:
                if not skip_flag:
                    image_queue.put((ret_val, frame))
                    skip_flag = True
                else:
                    skip_flag = False
            else:
                image_queue.put((ret_val, frame))
        else:
            break
        if not event_queue.empty():
            print('receive termination signal')
            break

    cap.release()
    print('RTSP closed')

def write_video(vid_writer, image_queue):
    is_terminated = False
    count = 0
    print('start writing...')
    while True:
        while True:
            if not image_queue.empty():
                ret_val, frame = image_queue.get()
                break
            else:
                time.sleep(0.001)

        if ret_val:
            vid_writer.write(frame)
            count += 1
        else:
            print(f'ret val is {ret_val}')
            print('terminating now')
            break

        if count % 120 == 0:
            print(f'recorded {count} frames')



def main():
    args = parse_args()
    time.sleep(args.countdown)

    # get FPS and image size
    cap = cv2.VideoCapture(args.rtsp_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    time.sleep(1)

    # start a thread to capture from rtsp
    image_queue = Queue()
    event_queue = Queue()
    cap_thread = threading.Thread(
        target=capture_from_rtsp,
        args=(args.rtsp_path, image_queue, event_queue, False)
    )
    cap_thread.start()

    vid_writer = cv2.VideoWriter(
        args.output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(width), int(height))
    )

    try:
        write_video(vid_writer, image_queue)
        event_queue.put(0)
        cap_thread.join()
    except Exception as error:
        event_queue.put(0)
        cap_thread.join()
        print(error)

if __name__ == '__main__':
    main()
