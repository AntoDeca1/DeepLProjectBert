from torch import nn
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def plot_results_neptune(plot_path, training=True):
    """
    Given a csv generated in the shape of the ones generated from read_accuracy_csv() ,realize plots
    proposed_results
    :param plot_path:
    :param training:
    :return:
    """
    df = pd.read_csv(plot_path, names=["C1", "C2", "C3"])

    fig = go.Figure()
    fig.add_trace(go.Line(x=df['C1'], y=df['C3'], mode='lines', name="Accuracy_Training"))
    # fig.add_trace(go.Line(x=df['index'], y=df['Validation_Accuracy'], mode='lines', name="Accuracy_Validation"))
    fig.update_layout(
        title='Training Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )
    fig.show()


def plot_results(plot_path, training=True):
    """
    Given a csv generated in the shape of the ones generated from read_accuracy_csv() ,realize
    proposed_results
    :param plot_path:
    :param training:
    :return:
    """
    df = pd.read_csv(plot_path)
    df = df.iloc[:30]

    fig = go.Figure()
    fig.add_trace(go.Line(x=df['index'], y=df['Training_Accuracy'], mode='lines', name="Accuracy_Training"))
    fig.add_trace(go.Line(x=df['index'], y=df['Validation_Accuracy'], mode='lines', name="Accuracy_Validation"))
    fig.update_layout(
        title='Accuracy Comparison',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Line(x=df['index'], y=df['Training_Losses'], mode='lines', name="Training_loss"))
    fig1.add_trace(go.Line(x=df['index'], y=df['Validation_Losses'], mode='lines', name="Validation_Loss"))
    fig1.update_layout(
        title='Losses Comparison',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    fig.show()
    fig1.show()


def read_accuracy_txt(path):
    """
    From the log taken from Kaggle or Colab extract the training and the validation losses/accuracies
    It can be used if we forget to keep track of them during the training process :)
    :param path:
    :return:
    """
    training_losses = []
    training_accuracies = []
    validation_accuracies = []
    validation_losses = []
    file = open(path, 'r')

    # Read the contents of the file
    file_contents = file.read()
    lines = file_contents.split("\n")  # List of lines
    lines_filterd = [line for line in lines if line.find("Epoch") != -1]
    for line in lines_filterd:
        line = line.split("\t")
        training_loss, validation_loss, training_accuracy, validation_accuracy = [float(el.split(":")[1]) for el in line
                                                                                  if
                                                                                  el.find("Training") != -1 or el.find(
                                                                                      "Validation") != -1]
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)
    file.close()

    df = pd.DataFrame(list(zip(training_losses, validation_losses, training_accuracies, validation_accuracies)),
                      columns=["Training_Losses", "Validation_Losses", "Training_Accuracy", "Validation_Accuracy"])
    print(
        f"Best Validation Accuracy: {max(validation_accuracies)}"
        f"\nBest Validation Epoch: {np.argmax(validation_accuracies)}"
        f"\nEpochs: {len(df)}"
        f"\nBest Training Accuracy {max(training_accuracies)}")
    df = df.reset_index()
    return df
