import math
import os
import os.path as path
import random
import Image
image_size = 250
samplesPerClass = 28
classesCount = 100

# paths
faceCordinatesPath = path.join('initial databases', 'faces', 'bbox.txt')
eyesCordinatesPath = path.join('initial databases', 'faces', 'landmarks.txt')
identityPath = path.join('initial databases', 'faces', 'identity.txt')
unsortedFaceFolder = path.join('initial databases', 'faces', 'pics')
temporaryFaceFolder = path.join('temporaryFaceDatabase')
allFaceFolder = path.join('final databases', 'faces', 'all')
testFaceFolder = path.join('final databases', 'faces', 'test')
trainFaceFolder = path.join('final databases', 'faces', 'train')
validaFaceFolder = path.join('final databases', 'faces', 'valid')

def create_face_database():
    identity = load_identities(identityPath, ",")
    allPersonNumbers = list(identity.values())
    if len(os.listdir(allFaceFolder)) ==0:
        for filename in os.listdir(unsortedFaceFolder):
            personNumber =identity.get(filename)
            if  allPersonNumbers.count(personNumber) >28:
                image = Image.open(os.path.join(unsortedFaceFolder, filename))
                imageName = filename.split('\\')[-1]
                if imageName in identity:
                    folderPath = os.path.join(allFaceFolder, str(personNumber))
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath)
                    if len(os.listdir(folderPath)) < 28:
                        image.save(os.path.join(folderPath, imageName))
    split_to_test_train_valid()

def load_identities(pathToFile, delimiter=" "):
    loadedIdentities = dict()
    with open(pathToFile, 'r') as f:
        for line in f:
            lineArray = line.split(delimiter)
            loadedIdentities[lineArray[0]] = int(lineArray[1])
    return loadedIdentities

def load_eye_coordinates(pathToFile, delimiter=" "):
    loadedCoordinates = dict()
    with open(pathToFile, 'r') as f:
        for line in f:
            lineArray = line.replace("    ", " ").replace("   ", " ").replace("  ", " ").replace("\n", "").split(
                delimiter)
            loadedCoordinates[lineArray[0]] = lineArray[1:]
    return loadedCoordinates


def crop_face(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.35, 0.35), dest_sz=(250, 250)):
    # https://bytefish.de/blog/aligning_face_images/
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = count_distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)

    return image

def count_distance(p1, p2):
    # https://bytefish.de/blog/aligning_face_images/
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    # https://bytefish.de/blog/aligning_face_images/
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

def transform_random_image(personNumber):
    image, imageName = get_random_image(str(personNumber))
    eyesCordinates = load_eye_coordinates(eyesCordinatesPath)
    eye_left = (int(eyesCordinates[imageName][0]), int(eyesCordinates[imageName][1]))
    eye_right = (int(eyesCordinates[imageName][2]), int(eyesCordinates[imageName][3]))
    rotated = crop_face(image, eye_left, eye_right)
    return rotated

def get_random_image(personNumber):
    imageName = random.choice(os.listdir(path.join(testFaceFolder,personNumber)))  # change dir name to whatever
    image = Image.open(path.join(testFaceFolder, personNumber, imageName))
    return image, imageName

def transform_image(image, imageName):
    eyesCordinates = load_eye_coordinates(eyesCordinatesPath)
    eye_left = (int(eyesCordinates[imageName][0]), int(eyesCordinates[imageName][1]))
    eye_right = (int(eyesCordinates[imageName][2]), int(eyesCordinates[imageName][3]))
    rotated = crop_face(image, eye_left, eye_right)
    return rotated

def file_image_to_folder(sourceFolder, newPersonName, image, imageName):
    sourceFolderFullPath = os.path.join(sourceFolder, newPersonName)
    if not os.path.exists(sourceFolderFullPath):
        os.makedirs(sourceFolderFullPath)
    image.save(os.path.join(sourceFolderFullPath, imageName))

def split_to_test_train_valid():
    if len(os.listdir(testFaceFolder))==0:
        newPersonName = 1
        trainSamplesCount = math.floor(samplesPerClass * 0.7)
        testSamplesCount = math.floor(samplesPerClass * 0.15)

        for index, oldPersonName in enumerate(os.listdir(allFaceFolder)):
            personalFolder = os.listdir(path.join(allFaceFolder, oldPersonName))
            if newPersonName > classesCount:
                break
            for idx, imageName in enumerate(personalFolder):
                image = Image.open(os.path.join(allFaceFolder, str(oldPersonName), imageName))
                if idx < trainSamplesCount:
                    file_image_to_folder(trainFaceFolder, str(newPersonName), image, imageName)
                elif idx < trainSamplesCount + testSamplesCount:
                    file_image_to_folder(testFaceFolder, str(newPersonName), image, imageName)
                else:
                    file_image_to_folder(validaFaceFolder, str(newPersonName), image, imageName)
            newPersonName = newPersonName + 1

def get_two_transformed_faces(firstPersonNumber, secondPersonNumber):
    firstPersonFace = transform_random_image(firstPersonNumber)
    secondPersonFace = transform_random_image(secondPersonNumber)
    return firstPersonFace, secondPersonFace


