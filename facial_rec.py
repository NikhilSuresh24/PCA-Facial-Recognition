from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from PCA import PCA
from tqdm import tqdm

lfw_dataset = fetch_lfw_people(min_faces_per_person=150)

X = lfw_dataset.data
y = lfw_dataset.target
target_names = lfw_dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

pca = PCA(target_names, 18)
pca.process_images(X_train, y_train)
pca.calculate_covariances()

print("train accuracy", pca.accuracy(X_train, y_train))
print("test accuracy:", pca.accuracy(X_test, y_test))
print(pca.num_ids)

# best_k = []
# best_acc = []
# num_trials = 100
# for trial in tqdm(range(num_trials)):
#     x = []
#     for i in range(100):
#         pca = PCA(target_names, K=i)
#         pca.process_images(X_train, y_train)
#         pca.calculate_covariances()
#         acc1 = pca.accuracy(X_train, y_train)
#         acc2 = pca.accuracy(X_test, y_test)
#         x.append(acc2)
#     best_k.append(x.index(max(x)))
#     best_acc.append(acc2)

# print("K:",  best_k, sum(best_k)/num_trials)
# print("ACC:", best_acc, sum(best_acc)/num_trials)
# for i in tqdm(range(100)):
# pca = PCA(target_names, K=i)
# pca.process_images(X_train, y_train)
# pca.calculate_covariances()
# acc1 = pca.accuracy(X_train, y_train)
# acc2 = pca.accuracy(X_test, y_test)
# x.append(acc1 + acc2)

# print(x.index(max(x)), max(x))
