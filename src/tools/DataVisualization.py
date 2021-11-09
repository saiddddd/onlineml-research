import traceback

import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt


class BasicHistogram:

    def __init__(self, input_data):

        self._X = input_data
        self._plt = plt

    def draw_distribution(self, **kwargs):
        self._plt.hist(self._X, **kwargs)
        self._plt.yscale('log')
        self._plt.grid()

    def set_xlabel(self, x_label):
        self._plt.xlabel(x_label)

    def set_ylabel(self, y_label):
        self._plt.ylabel(y_label)

    def set_title(self, title):
        self._plt.title(title)

    def save_fig(self, saving_location):
        self._plt.savefig(saving_location)

class HistogramCompare:

    def __init__(self, nominal_data, comparing_data):
        """
        Drawing distribution and compare distribution
        """

        self._nominal_data_dist = nominal_data
        self._comparing_data_dist = comparing_data

    def draw_nominal_dist(self, data_name='X', binning_size=50, density=True, facecolor='c', alpha=0.75):
        plt.hist(self._nominal_data_dist, binning_size, density=density, facecolor=facecolor, alpha=alpha)
        plt.xlabel(data_name)
        if density is False:
            plt.ylabel('Statistics')
        else:
            plt.ylabel('A.U.')
        plt.title('Nominal data distribution')
        plt.grid(True)
        plt.show()

    def draw_nominal_comparing_dist(self, data_name='X', binning_size=50, density=True, nominal_facecolor='c', comparing_facecolor='r', alpha=0.75):

        plt.hist(self._nominal_data_dist, binning_size, density=density, facecolor=nominal_facecolor, alpha=alpha, label='nominal')
        plt.hist(self._comparing_data_dist, binning_size, density=density, facecolor=comparing_facecolor, alpha=alpha, label='compared')
        plt.xlabel(data_name)
        if density is False:
            plt.ylabel('Statistics')
        else:
            plt.ylabel('A.U.')
        plt.title('{} Distribution Comparison'.format(data_name))
        plt.grid(True)
        plt.legend()
        plt.show()


class TrendPlot:
    def __init__(self):
        pass

    def plot_trend(self, *args, label='unnammed trend'):
        plt.plot(*args, label=label)

    def plot_trend_with_error_bar(self, *args, y_err, label='unnammed trend'):
        plt.errorbar(*args, yerr=y_err, fmt='o', markersize=4, capsize=2, label=label)

    def plot_trend_with_error_band(self, *args, y_err, label='unnammed trend'):

        def list_err_band_list(input_nominal, input_err):
            if len(input_nominal) == len(input_err):
                list_length = len(input_nominal)
                output_upper_list = []
                output_lower_list = []
                for i in range(list_length):
                    output_upper_list.append(input_nominal[i] + input_err[i])
                    output_lower_list.append(input_nominal[i] - input_err[i])
                return output_upper_list, output_lower_list
            else:
                raise IndexError("Input list a and list b is not in the same size")

        arg = [arg for arg in args]
        x = arg[0]
        y = arg[1]
        try:
            yerr_upper, yerr_lower = list_err_band_list(y, y_err)
            plt.plot(*args)
            plt.fill_between(x, yerr_upper, yerr_lower, alpha=0.5, label=label)

        except IndexError:
            traceback.print_exception()

    def show(self):
        plt.show()


    def save_fig(self, title=None, x_label='data list sequence', y_label='unnamed data', save_fig_path='./output_fig.pdf'):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.tick_params(axis="y", direction="inout")
        plt.tick_params(labelleft=True, labelright=True)
        plt.grid(axis='y', alpha=.7, linestyle=":")
        plt.savefig(save_fig_path)


class CalibrationPlot:

    def __init__(self):
        mpl.style.use('fivethirtyeight')

        plt.figure(figsize=(10, 10))
        self._ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        self._ax2 = plt.subplot2grid((3, 1), (2, 0))

        self._ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrates")


    def add_calibration_curve(self, mean_predicted_value, fraction_of_positives, label="unnamed model"):
        self._ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)

    def add_histogram(self, prob_pos, name):
        self._ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype='step', lw=2)

    def save_fig(self, title="Calibration plots  (reliability curve)"):
        self._ax1.set_ylabel("Fraction of positives")
        self._ax1.set_ylim([-0.05, 1.05])
        self._ax1.legend(loc='lower right')
        self._ax1.set_title(title)

        self._ax2.set_xlabel("Mean predicted value")
        self._ax2.set_ylabel("Count")
        self._ax2.legend(loc="upper center", ncol=2)
        plt.tight_layout()


        plt.savefig('./output_calibration.pdf')

