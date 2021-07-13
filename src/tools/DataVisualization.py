import traceback

import matplotlib.pyplot as plt


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


    def save_fig(self, title=None, x_label='data list sequence', y_label='unnamed data'):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.tick_params(axis="y", direction="inout")
        plt.tick_params(labelleft=True, labelright=True)
        plt.grid(axis='y', alpha=.7, linestyle=":")
        plt.savefig('./output_trend_plot_functional.pdf')


