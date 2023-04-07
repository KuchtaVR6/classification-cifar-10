import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import my_utils as mu
import os
import shutil


def fetch_data_cifar(batch_size, num_workers, resize=None):
    """ fetch data and return dataLoaders based on this data """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    loaded_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=trans)

    loaded_test = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=trans)

    return (data.DataLoader(loaded_train, batch_size, shuffle=True,
                            num_workers=num_workers),
            data.DataLoader(loaded_test, batch_size, shuffle=False,
                            num_workers=num_workers))


class CoefficientProducer(torch.nn.Module):
    # input_dimensions - an array or torch.Size() with the dimensions of the
    #                     that will be given
    def __init__(self, input_dimensions, input_channels, batch_size, conv_count):
        super(CoefficientProducer, self).__init__()

        # take the special average using AvgPool where the kernel size
        # is the largest dimension (barring the channels hence [1:])

        self.special_average = nn.AvgPool2d(max(input_dimensions[1:]), padding=0)

        # implementing the Linear layer
        self.arguments_producer = nn.Linear(input_channels, conv_count)

        # non-linear activation function
        self.argument_activation = nn.LeakyReLU()

        self.input_channels = input_channels
        self.batch_size = batch_size

    def forward(self, x):
        out = self.special_average(x)
        # output is batch_size x channel x 1 x 1, as such it ought to flattened
        flattened = out.view(out.size()[0], -1)
        out = self.arguments_producer(flattened)
        out = self.argument_activation(out)
        return out


class BackBone(torch.nn.Module):
    # input_dimensions - an array or torch.Size() with the dimensions of the
    #                     that will be given
    def __init__(self, input_dimensions, input_channels, output_channels,
                 conv_count, batch_size, kernel_size=3, avg_kernel=3, avg_stride=2):

        super(BackBone, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dimensions = input_dimensions
        self.conv_count = conv_count
        self.avg_kernel = avg_kernel
        self.avg_stride = avg_stride
        self.coefficient_producer = CoefficientProducer(input_dimensions, input_channels, batch_size, conv_count)

        # additions to the model
        self.activation = nn.ReLU()
        self.pooling = nn.AvgPool2d(avg_kernel, stride=avg_stride, padding=(avg_kernel - 1) // 2)

        for i in range(conv_count):
            # the output is equal to conv_count as we want an output for each of the
            # convolutions layer

            # padding is set to (kernel_size - 1)//2 to maintain the size

            self.add_module('conv' + str(i), nn.Conv2d(input_channels, output_channels, kernel_size,
                                                       padding=(kernel_size - 1) // 2))

    def forward(self, x):
        # get the coefficients
        coefficient = self.coefficient_producer.forward(x)

        # creating a tensor of zeros where the output will be added to
        # size is based on x.size() with number of channels set to the number of
        # output channels
        size_overall = list(x.size())[:1] + [self.output_channels] + list(x.size())[2:]
        overall = torch.zeros(size_overall)

        index = 0
        for i in range(self.conv_count):
            out = self._modules['conv' + str(i)](x)
            # for each from the batch
            for inner_index in range(len(coefficient[:, index])):
                # adding the output of convolution multiplied by the coefficient
                overall[inner_index] = coefficient[inner_index, index] * out[inner_index]
            index += 1

        overall = self._modules['conv0'](x)

        # additions to the model
        overall = self.activation(overall)
        overall = self.pooling(overall)

        return overall

    def get_output_dimensions(self):
        return [
            self.output_channels,
            self.input_dimensions[1] // self.avg_stride,
            self.input_dimensions[2] // self.avg_stride
        ]


class MyNet(torch.nn.Module):

    def __init__(self,
                 num_backbones,
                 conv_count,
                 channels,
                 # dimension of each batch
                 input_dimensions,
                 num_outputs=10):
        if len(channels) != num_backbones + 1:
            raise "Wrong number of channels, as compared to num_backbones"

        super(MyNet, self).__init__()
        self.num_inputs = np.prod(input_dimensions[1:])
        self.input_dimensions = input_dimensions
        self.channels = channels
        self.num_outputs = num_outputs
        self.num_backbones = num_backbones
        matrix_dimensions = self.input_dimensions[1:]

        # additions to the model
        self.activate_each = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)

        for index in range(num_backbones):
            this_back_bone = BackBone(
                matrix_dimensions,
                channels[index],
                channels[index + 1],
                conv_count,
                self.input_dimensions[0]
            )
            self.add_module('back' + str(index), this_back_bone)

            matrix_dimensions = this_back_bone.get_output_dimensions()

        self.special_average = nn.AvgPool2d(max(matrix_dimensions[1:]), padding=0)

    def forward(self, x):
        for i in range(self.num_backbones):
            x = self._modules['back' + str(i)](x)
            # additions to the model
            x = self.activate_each(x)
            x = self.dropout(x)

        x = self.special_average(x)
        return x.view(self.input_dimensions[0], -1)


def train_ch3_once(net, train_iter, test_iter, loss, updater):
    train_metrics = mu.train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = mu.evaluate_accuracy(net, test_iter)
    return [train_metrics[0], train_metrics[1], test_acc]


def int_input(question, set_default=None):
    while True:
        try:
            res = input(question)
            if set_default is not None and len(res) == 0:
                res = set_default
            else:
                res = int(res)
            break
        except ValueError:
            print("Integer numbers only please!")
    return res


def float_input(question, set_default=None):
    while True:
        try:
            res = input(question)
            if set_default is not None and len(res) == 0:
                res = set_default
            else:
                res = float(res)
            break
        except ValueError:
            print("Floating point numbers only please!")
    return res


class ReportFile:

    def __init__(self, name, note=""):
        self.name = name
        self.note = note
        self.current_accuracy = 0
        self.current_epoch = 0
        self.metrics = []
        self.dimensions = [[], 0]
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 16

    def read_file(self, file):
        index = 0
        for line in file:
            text = line.replace("\n", "")
            if index == 0:
                self.note = text
            elif index == 1:
                self.current_accuracy = float(text)
            elif index == 2:
                self.current_epoch = int(text)
            elif index == 3:
                split = text.split(",")
                for i in range(len(split)):
                    split[i] = int(split[i])
                self.dimensions = [split[1:], split[0]]
            elif index == 4:
                self.learning_rate = float(text)
            elif index == 5:
                self.weight_decay = float(text)
            elif index == 6:
                self.batch_size = int(text)
            else:
                split = text.split(",")
                split[0] = int(split[0])
                split[1] = float(split[1])
                split[2] = float(split[2])
                split[3] = float(split[3])
                self.metrics.append(split)
            index += 1

    def set_dimensions(self):
        if self.dimensions[1] == 0:
            print("         __________________________________________________________________")
            print("         Please select the models dimensions.")
            self.dimensions[1] = int_input("                  Input the number of convolutional layers per" + \
                                           " backbone (default setting -> 1): ", set_default=1)
            print("         Now select the number of output channels for each backbone (type 0 to end): ")
            print("         Note the last 10 channel layer is included by default!")
            index = 0
            self.dimensions[0].append(3)  # 3 colours
            channels = int_input("                  Backbone number " + str(index) + " output channels: ")
            while channels != 0:
                index += 1
                self.dimensions[0].append(channels)
                channels = int_input("                  Backbone number " + str(index) + " output channels: ")

            self.dimensions[0].append(10)  # 10 classes

    def set_params(self):
        print("         __________________________________________________________________")
        lr = float_input(("                  Please input the desired learning rate (currently {}) " + \
                         "(default setting -> 0.001): ").format(self.learning_rate), set_default=0.001)

        wd = float_input(("                  Please input the desired weight decay (currently {}) " + \
                         "(default setting -> 0.0001): ").format(self.weight_decay), set_default=0.0001)

        bs = int_input(("                  Please input the desired batch size (currently {}) " + \
                        "(default setting -> 16): ").format(self.batch_size), set_default=16)

        self.learning_rate = lr
        self.weight_decay = wd
        self.batch_size = bs

    def add_metrics(self, metrics, saved=""):
        self.current_epoch += 1
        self.current_accuracy = metrics[2]
        if len(saved) > 0:
            self.metrics.append([self.current_epoch] + metrics + ["(saved separately as {})".format(saved)])
        else:
            self.metrics.append([self.current_epoch] + metrics)

    def serialise(self):
        metrics_output = ""
        for metric in self.metrics:
            metrics_output += str(metric[0]) + "," + str(metric[1]) + "," + str(metric[2]) + "," + str(metric[3])
            if len(metric) > 4:
                metrics_output += "," + str(metric[4])
            metrics_output += "\n"
        dimensions_output = str(self.dimensions[1]) + ","
        for dimension in self.dimensions[0]:
            dimensions_output += str(dimension) + ","
        dimensions_output = dimensions_output[:-1]
        return self.note + "\n" + str(self.current_accuracy) + "\n" + str(self.current_epoch) + "\n" + \
            dimensions_output + "\n" + str(self.learning_rate) + "\n" + str(self.weight_decay) + "\n" + \
            str(self.batch_size) + "\n" + metrics_output

    def serialise_csv(self):
        metrics_output = ""
        for metric in self.metrics:
            metrics_output += str(metric[0]) + "," + str(metric[1]) + "," + str(metric[2]) + "," + str(metric[3])
            metrics_output += "\n"
        return metrics_output

    def get_basic_details(self):
        note = ""
        if len(self.note) > 0:
            note = " (NOTE: " + self.note + ")"

        return self.name + note + ", currently on epoch number: " + str(self.current_epoch) + ", with accuracy: " + \
            str(round(self.current_accuracy * 100, 2)) + "% and the channels used are: " + str(self.dimensions[0]) + \
            " with " + str(self.dimensions[1]) + " convolutional layer(s) per backbone. Learning rate = " + \
            str(self.learning_rate) + ", weight_decay = " + str(self.weight_decay) + " and batch size = " + \
            str(self.batch_size) + "."


def main():
    models = []
    index = 2

    print("====================== WELCOME =================")
    print("Please note: To select the default value simply press 'ENTER' or 'RETURN' with a blank input.")
    print("__________________________________________________________________")

    print("Options:")
    print("0: Create a new model.")

    fork_printed = False

    for folder_name in os.listdir('files'):
        for filename in os.listdir('files/' + folder_name):
            if filename == "main.report":
                if not fork_printed:
                    fork_printed = True
                    print("1: Fork an existing model.")
                with open('files/' + folder_name + '/' + filename) as file:
                    new_report = ReportFile(folder_name)
                    new_report.read_file(file)
                    models.append(new_report)
                    print(index, ":", new_report.get_basic_details())
                    index += 1

    continuation = True
    report = None
    choice = 0

    print("         __________________________________________________________________")

    while continuation:
        choice = int_input("         Select the model to train: ")
        continuation = False
        if choice == 0:
            name = input("         Creating a new model, please select a non-empty name: ")
            note = input("         Please type in any relevant notes (can be left empty): ")
            report = ReportFile(name, note=note)
            report.set_dimensions()
            report.set_params()
            try:
                os.mkdir("files/" + name)
                os.mkdir("files/" + name + "/autosaves")
            except FileExistsError:
                pass
        elif choice == 1:
            cont_inner = True
            while cont_inner:
                print("         _________________________________________________________")
                cont_inner = False
                choice_2 = int_input("                  Which model you would like to fork (use the list above): ")
                if 1 < choice_2 < len(models) + 2:
                    report = models[choice_2 - 2]
                else:
                    print("         Invalid input")
                    continuation = True
            name = input("                  Please select a non-empty name: ")
            shutil.copytree("files/" + report.name, "files/" + name)
            report.name = name
        elif choice < len(models) + 2:
            report = models[choice - 2]
        else:
            print("Invalid input")
            continuation = True

    model_path = "files/" + report.name + "/main.model"
    report_path = "files/" + report.name + "/main.report"
    csv_report_path = "files/" + report.name + "/report.csv"
    autosave_path = "files/" + report.name + "/autosaves/"

    epoch = report.current_epoch + 1

    autosave_type = 3

    autosave_epochs_interval = 5
    old_autosave_epochs = (report.current_epoch // 5) * 5
    autosave_improvement_interval = 0.05
    old_autosave_accuracy = (report.current_accuracy // 0.05) * 0.05

    print("__________________________________________________________________")

    print("Set up termination policy:")
    print("There are two rules: Terminate after accuracy reaches X%, and Terminate after epochs (overall) reach X.")
    print("Whatever happens first takes precedence.")
    print("Please input values: (example: 95 (%), and 50 (epochs))")

    termination_accuracy = int_input("         Terminate after accuracy reaches (default setting -> 100): ",
                                     set_default=100)
    termination_epochs = int_input("         Terminate after epochs (overall) reach (default setting -> 10): ",
                                   set_default=10)

    print("__________________________________________________________________")

    continuation = True
    while continuation:
        print("Adjustment Menu:")
        print("0: Adjust the learning rate, weight decay or batch size")
        print("1: Replace the note")
        print("2: Display details of the model")
        print("3: Save settings (saved automatically regardless on each autosave and termination)")
        print("4: Adjust the autosave policy, currently ", end="")
        if autosave_type == 0:
            print("no autosave")
        elif autosave_type == 1:
            print("autosave overwriting the main file")
        elif autosave_type == 2:
            print("autosaves in a separate folder")
        elif autosave_type == 3:
            print("autosaves in a separate folder + overwriting the main file (recommended)")
        print("5: Continue without further adjustments")

        choice_3 = int_input("         Select the option (default option -> 5): ", set_default=5)

        if choice_3 == 0:
            report.set_params()
        elif choice_3 == 1:
            print("         _________________________________________________________")
            if len(report.note) == 0:
                new_note = input("                  Please input the new note: ")
            else:
                new_note = input("                  Please input the new note (old note: {}): ".format(report.note))
            report.note = new_note
        elif choice_3 == 2:
            print(report.get_basic_details())
        elif choice_3 == 3:
            with open(report_path, "w") as file:
                file.write(report.serialise())
            print("Saved")
        elif choice_3 == 4:
            print("         _________________________________________________________")
            print("                  0: No autosave")
            print("                  1: Autosave overwriting the main file")
            print("                  2: Autosave in a separate folder")
            print("                  3: Autosave in a separate folder + overwriting the main file (recommended)")
            typed = int_input("                  Type the number of the selected option: ")
            if typed == 1 or typed == 2 or typed == 0 or typed == 3:
                autosave_type = typed
            else:
                print("Invalid input")
        elif choice_3 == 5:
            continuation = False
        else:
            print("Invalid number")

        print("__________________________________________________________________")

    print("====================== SET UP COMPLETE ===============")
    print(report.get_basic_details())
    print("====================== PREPARING =====================")
    # loading the train and test dataloaders

    num_workers = 8
    train_loader, test_loader = fetch_data_cifar(report.batch_size, num_workers)

    # getting the first batch
    x, y = next(iter(train_loader))

    loss = nn.CrossEntropyLoss()

    this_net = MyNet(len(report.dimensions[0]) - 1, report.dimensions[1], report.dimensions[0], x.size())
    optimizer = torch.optim.Adam(this_net.parameters(), lr=report.learning_rate, weight_decay=report.weight_decay)

    if choice != 0:
        this_net.load_state_dict(torch.load(model_path))
        print("Model successfully loaded!")

    print("====================== PREPARING COMPLETED ===========")
    print("====================== TRAINING ======================")

    print("epoch, loss, train_accuracy, test_accuracy")

    while epoch <= termination_epochs and report.current_accuracy < termination_accuracy / 100:
        print(epoch, end=", ")
        result = train_ch3_once(
            this_net,
            train_loader,
            test_loader,
            loss,
            optimizer,
        )

        print(str(result[0]) + ", " + str(result[1]) + ", " + str(result[2]))

        to_be_saved = False
        autosave_name = "autosave"

        if autosave_type > 0:
            if epoch - old_autosave_epochs >= autosave_epochs_interval:
                autosave_name += "Epo" + str(report.current_epoch + 1)
                old_autosave_epochs = old_autosave_epochs + autosave_epochs_interval
                to_be_saved = True
            if result[2] - old_autosave_accuracy >= autosave_improvement_interval:
                autosave_name += "Acc" + str(int(result[2] * 100))
                old_autosave_accuracy = old_autosave_accuracy + autosave_improvement_interval
                to_be_saved = True
            autosave_name += ".model"
            if to_be_saved:
                if autosave_type == 1 or autosave_type == 3:
                    torch.save(this_net.state_dict(), model_path)
                if autosave_type == 2 or autosave_type == 3:
                    torch.save(this_net.state_dict(), autosave_path + autosave_name)

        if to_be_saved and (autosave_type == 2 or autosave_type == 3):
            report.add_metrics(result, saved=autosave_name)
        else:
            report.add_metrics(result)

        if to_be_saved:
            with open(report_path, "w") as file:
                file.write(report.serialise())

        epoch += 1

    print("====================== TRAINING TERMINATED - SAVING...")
    torch.save(this_net.state_dict(), model_path)
    with open(report_path, "w") as file:
        file.write(report.serialise())
    with open(csv_report_path, "w") as file:
        file.write(report.serialise_csv())
    print("====================== FILES SAVED, GOOD BYE! ========")


if __name__ == '__main__':
    main()
