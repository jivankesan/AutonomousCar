import cv2
import asyncio
from open_gopro import gopro_wired


async def main():
    gopro = gopro_wired.WiredGoPro()
    await gopro.open()
    print("success")

    # Check the correct method name for starting the webcam preview
    url = await gopro.http_command.set_preview_stream(mode=1)  # Replace 'start_preview' with the correct method

    cap = cv2.VideoCapture(url.identifier)

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('gopro', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    print("exit success")


asyncio.run(main())
