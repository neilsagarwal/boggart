import pandas as pd
from PIL import Image


def trajectory_process():
    # CSV 파일 경로
    file_path = 'minimum_frame_set.csv'

    # CSV 파일 로드
    data = pd.read_csv(file_path)
    data = data.drop_duplicates(subset='FrameNumber')

    # 'FrameNumber' 컬럼을 기준으로 데이터 정렬
    sorted_data = data.sort_values(by='FrameNumber')

    # 정렬된 데이터 출력
    print(sorted_data)
    sorted_data.to_csv("sorted_unique.csv", index=False)

# frame에서 필요없는 부분 절대 object 나타나지 않을 부분 crop
def frame_analysis():
    # 이미지 파일 열기
    img = Image.open("/home/kth/rva/jackson_image.png")

    # 자르고자 하는 영역 지정: (x1, y1, x2, y2)
    crop_area = (0, 0, 1920, 600)  # 여기서 좌표는 예시이므로 실제 필요한 값으로 조정하세요.

    # 영역 자르기
    cropped_img = img.crop(crop_area)

    # 잘라낸 이미지 저장
    cropped_img.save("/home/kth/rva/jackson_cropped_image.jpg")



trajectory_process()