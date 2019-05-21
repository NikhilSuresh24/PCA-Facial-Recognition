import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, target_names, K=100,  max_images_per_person=50):
        self.K = K
        self.faces = []
        self.eigenvectors = []
        self.avg_faces = []
        self.target_names = target_names
        self.ids = []
        self.num_ids = len(target_names)
        self.omegas = np.empty((self.num_ids, self.K))
        self.max_images = max_images_per_person

    def process_images(self, images, labels):
        for name in range(self.num_ids):
            face_locations = np.where(labels == name)
            faces = images[face_locations]
            # if faces.shape[0] <= self.max_images:
            self.faces.append(faces)
            #     self.ids.append(name)
            # else:
            #     self.num_ids -= 1

    def calculate_covariances(self):
        for idx, images in enumerate(self.faces):
            num_images, img_shape = images.shape

            # print("------IMG", images.shape)
            avg_face = np.sum(images, 0)/images.shape[0]
            self.avg_faces.append(avg_face)
            # print("avg", avg_face.shape)
            centered_faces = np.array([img - avg_face for img in images]).T
            # print("CENter", centered_faces.shape)
            L = np.matmul(centered_faces.T, centered_faces)
            # print("L", L.shape)
            evals, evecs = np.linalg.eig(L)

            # print("EIG", evals.shape, evecs.shape)
            c_evecs = np.empty((self.K, img_shape))
            for component in range(self.K):
                # print(evecs.shape, centered_faces.shape)
                c_evecs[component] = np.matmul(
                    evecs[component % num_images], centered_faces.T)

            self.eigenvectors.append(c_evecs)

            plt.imshow(c_evecs[0].reshape(62, -1), cmap='gray')
            plt.title("{}'s Eigenface".format(self.target_names[idx]))
            plt.savefig(
                "./eigenfaces/{}'s Eigenface".format(self.target_names[idx]))

            # Sample Image
            plt.imshow(images[0].reshape(62, -1), cmap='gray')
            plt.title("{} Sample Image".format(self.target_names[idx]))
            plt.savefig(
                "./images/{} Sample Image".format(self.target_names[idx]))

            # Average Face
            plt.imshow(avg_face.reshape(62, -1), cmap='gray')
            plt.title("{} Average Image".format(self.target_names[idx]))
            plt.savefig(
                "./avgfaces/{}'s Average Image".format(self.target_names[idx]))

            # Centered Face
            plt.imshow(centered_faces.T[0].reshape(62, -1), cmap='gray')
            plt.title("{} Centered Image".format(self.target_names[idx]))
            plt.savefig(
                "./centeredfaces/{}'s Centered Image".format(self.target_names[idx]))

            omega = np.empty(self.K)
            for component in range(self.K):
                omega[component] = np.matmul(c_evecs[component],
                                             centered_faces.T[component % num_images])
            # print(omega, omega.shape)
            self.omegas[idx] = omega

    # def display(self):

    def test_image(self, image):
        dists = np.empty((self.num_ids))
        # print(self.eigenvectors)
        for i in range(self.num_ids):
            eigenvecs = self.eigenvectors[i]
            avg_face = self.avg_faces[i]
            omega = np.empty(self.K)
            centered_face = image - avg_face
            for component in range(self.K):
                # print("\nCHECK", image.shape, avg_face.shape, image - avg_face)
                omega[component] = np.matmul(
                    eigenvecs[component].T, image - avg_face)
                # omega[component] = np.matmul(
                #     (image - avg_face).T, eigenvecs[component])
                # print(omega[component])
            # print("SUB", omega-self.omegas[i],
            #       np.linalg.norm(omega-self.omegas[i]))
            dists[i] = np.linalg.norm(omega - self.omegas[i])

        # print(dists, self.num_ids)
        # print(dists)
        return np.argmin(dists)

    def accuracy(self, images, labels):
        class_count = {}
        correct_count = {}
        correct_counter = 0
        total_counter = 0
        # print(self.ids)
        for image, label in tqdm(zip(images, labels)):
            # print(images, labels)
            # if label not in self.ids:
            #     continue
            total_counter += 1
            pred = self.test_image(image)
            if pred not in class_count:
                class_count[pred] = 1
            else:
                class_count[pred] += 1
            if pred == label:
                if pred not in correct_count:
                    correct_count[pred] = 1
                else:
                    correct_count[pred] += 1
                correct_counter += 1

            self.save_test(image, pred, label)
        print("Guess Tracker: ", class_count)
        print("Correct per Class: ", correct_count)
        # print(pred, label, correct_counter, total_counter)

        acc = correct_counter/total_counter
        # print(acc > 0.5 or self.num_ids != 2)
        return acc if (acc > 0.5 or self.num_ids != 2) else 1 - acc

    def save_test(self, image, pred, label):
        file_path = "./tests/correct/" if pred == label else "./tests/incorrect/"
        test_num = 0
        if os.listdir(file_path) != []:
            test_num = max([int(fn.split('test')[1].split(".")[0])
                            for fn in os.listdir(file_path)]) + 1

        plt.imshow(image.reshape(62, -1), cmap='gray')
        plt.title("pred: {} \n actual: {}".format(
            self.target_names[pred], self.target_names[label]))
        plt.savefig(
            file_path + "test" + str(test_num))
