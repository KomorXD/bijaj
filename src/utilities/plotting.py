import matplotlib.pyplot as plt
import torch
import random


def plot_and_save_results(test_visualize, test_dataset, model, output_path, count):
    random_indices = random.sample(range(0, len(test_dataset)), count)

    for i in range(count):
        n = random_indices[i]

        image_vis = test_visualize[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        plt.figure(figsize=(16, 9))

        plt.subplot2grid((2, 4), (0, 0), rowspan=2)
        plt.xticks([])
        plt.yticks([])
        plt.title("Original image")
        plt.imshow(image_vis)

        mask_labels = ['animal', 'masking bg', 'non-masking bg', 'eye-catching fg']

        for j in range(3, 7):
            plt.subplot(2, 6, j)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Truth {mask_labels[j - 3]}")
            plt.imshow(gt_mask[j - 3, ...].squeeze())

        for j in range(9, 13):
            plt.subplot(2, 6, j)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Pred. {mask_labels[j - 9]}")
            plt.imshow(pr_mask[j - 9, ...].squeeze())

        plt.savefig(f"{output_path}/result-{i}.png")


def plot_and_save_training_data(iou_scores, loss_points, output_path):
    epoches = list(range(0, len(iou_scores["train"])))

    plt.figure(figsize=(18, 9))

    plt.subplot(1, 2, 1)
    plt.plot(epoches, iou_scores["train"], color="blue", label="Train IoU score")
    plt.plot(epoches, iou_scores["valid"], color="red", label="Valid IoU score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoches, loss_points["train"], color="blue", label="Train loss")
    plt.plot(epoches, loss_points["valid"], color="red", label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.savefig(f"{output_path}/training_data.png")
