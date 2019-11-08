import cv2
import sys
import os

#images_parent_path = '.\\data\\celeb\\img_align_celeba\\img_align_celeba\\face'
images_parent_path = '.\\data\\celeb\\img_align_celeba\\img_align_celeba\\img_align_celeba'
save_path = '.\\data\\celeb\\cropped_celeba'

class FaceCropper(object):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, full_save_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            #lastimg = cv2.resize(faceimg, (32, 32))
            i += 1
            cv2.imwrite(full_save_path, faceimg)



detecter = FaceCropper()
#detecter.generate('.\\data\\000001.jpg', False)
num = 0
for file in os.listdir(images_parent_path):
    filename = os.path.join(images_parent_path, file)
    detecter.generate(filename,os.path.join(save_path,file),False)
    print(num,file)
    num += 1
