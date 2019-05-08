import face_utilities
import face_verification
import random

def run_multimodal_system():
    firstPersonNr = 49
    secondPersonNr = 87

    #Show biometrics and result to user
    _, _, firstPersonFace, secondPersonFace = face_verification.compare(firstPersonNr, secondPersonNr)
    # todo: show eye and ear biometrics and result to user

    # todo: find out how to calculate z-score

def main():
    #Create multimodal database
    # face_utilities.create_face_database()
    # todo: add eye and ear methods to create 100 person folders (1-100) splitted to train(70%), test(15%) and valid(15%) data

    # Multimodal system
    run_multimodal_system()

    # Evaluate system success
    face_verification.print_ROC()


main()