import face_recognition
from PIL import Image, ImageDraw


def face_rec():
    gal_face_img = face_recognition.load_image_file("img/gal1.jpg")
    gal_face_location = face_recognition.face_locations(gal_face_img)

    justice_league_img = face_recognition.load_image_file("img/justice_league_actors.jpg")
    justice_league_faces_locations = face_recognition.face_locations(justice_league_img)

    # print(gal_face_location)
    # print(justice_league_faces_locations)
    # print(f"Found {len(gal_face_location)} face(s) in this image")
    # print(f"Found {len(justice_league_faces_locations)} face(s) in this image")

    pil_img1 = Image.fromarray(gal_face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for(top, right, bottom, left) in gal_face_location:
        draw1.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw1
    pil_img1.save("img/new_gal1.jpg")

    pil_img2 = Image.fromarray(justice_league_img)
    draw2 = ImageDraw.Draw(pil_img2)

    for(top, right, bottom, left) in justice_league_faces_locations:
        draw2.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw2
    pil_img2.save("img/new_justice_league.jpg")


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for face_location in faces_locations:
        top, right, bottom, left = face_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/{count}_face_img.jpg")
        count += 1

    return f"Found {count} face(s) in this photo"


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    # print(img1_encodings)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    # print(result)

    if result[0]:
        print("Welcome to the club! :*")
    else:
        print("Sorry, not today... Next!")


def main():
    # face_rec()
    # print(extracting_faces("img/justice_league_actors.jpg"))
    compare_faces("img/gal1.jpg", "img/gal2.jpg")


if __name__ == '__main__':
    main()
