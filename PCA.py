import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.image import imread


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

    def read_dir(self, dir_path):
        image_paths = os.listdir(dir_path)
        num_images = len(image_paths)
        img_shape = self.read_img(dir_path + '/' + image_paths[0]).shape
        images = np.empty((num_images, img_shape[0], img_shape[1]))
        for idx, img_path in enumerate(image_paths):
            images[idx] = self.read_img(dir_path + '/' + img_path)
        return images

    def read_img(self, image):
        img = imread(image)
        img = np.sum(img, 2)/img.shape[2]
        plt.imshow(img, cmap='gray')
        # plt.show()
        return img

    def construct_eigenfaces(self, name, images):
        print(images.shape)

        num_images, h, w = images.shape
        avg_face = np.sum(images, 0)/images.shape[0]
        centered_faces = np.array([img - avg_face for img in images]).T

        L = np.matmul(centered_faces.T, centered_faces)
        evals, evecs = np.linalg.eig(L)

        c_evecs = np.empty((self.K, h, w))
        for component in range(self.K):
            c_evecs[component] = np.matmul(
                evecs[component % num_images], centered_faces.T)

        plt.imshow(c_evecs[0], cmap='gray')
        plt.title("{}'s Eigenface".format(name))
        plt.savefig(
            "./eigenfaces/{}'s Eigenface".format(name))

    def calculate_covariances(self):
        for idx, images in enumerate(self.faces):
            num_images, img_shape = images.shape

            avg_face = np.sum(images, 0)/images.shape[0]
            self.avg_faces.append(avg_face)
            centered_faces = np.array([img - avg_face for img in images]).T
            print(centered_faces.shape)
            L = np.matmul(centered_faces.T, centered_faces)
            evals, evecs = np.linalg.eig(L)

            c_evecs = np.empty((self.K, img_shape))
            for component in range(self.K):
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
            self.omegas[idx] = omega

    # def display(self):

    def test_image(self, image):
        dists = np.empty((self.num_ids))
        for i in range(self.num_ids):
            eigenvecs = self.eigenvectors[i]
            avg_face = self.avg_faces[i]
            omega = np.empty(self.K)
            for component in range(self.K):
                omega[component] = np.matmul(
                    eigenvecs[component].T, image - avg_face)

            dists[i] = np.linalg.norm(omega - self.omegas[i])

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
                    self.save_test(image, pred, label)
                else:
                    correct_count[pred] += 1
                correct_counter += 1

            # self.save_test(image, pred, label)
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
