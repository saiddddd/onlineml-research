from matplotlib import pyplot as plt
from matplotlib import dates as mdates


class PredictionProbabilityDist:
    
    def __init__(self, pred_proba_result_list, target_list):
        
        # plt.clf()
        # plt.cla()
        
        self._pred_proba_result_list = pred_proba_result_list
        self._target_list = target_list
        
    def draw_proba_dist_by_true_false_class_seperated(self):

        topic_str = ''
        
        pred_proba_result_true_class = self._pred_proba_result_list[self._target_list == 1]
        pred_proba_result_false_class = self._pred_proba_result_list[self._target_list == 0]
        
        fig = plt.figure(figsize=(14, 4))
        fig.suptitle('{}pred_proba_distribution'.format(topic_str))
        plt.subplot(131)
        plt.hist(pred_proba_result_true_class, bins=50, alpha=0.5, label='Y True')
        plt.hist(pred_proba_result_false_class, bins=50, alpha=0.5, label='Y False')
        plt.yscale('log')
        plt.title('stacking prediction proba in both class')
        plt.xlabel('pred proba')
        plt.ylabel('statistics')
        plt.grid()
        plt.legend()
        plt.subplot(132)
        plt.hist(pred_proba_result_true_class, bins=50)
        plt.yscale('log')
        plt.title('Y True class prediction proba. dist.')
        plt.xlabel('pred proba')
        plt.ylabel('statistics')
        plt.grid()
        plt.subplot(133)
        plt.hist(pred_proba_result_false_class, bins=50)
        plt.yscale('log')
        plt.title('Y False class prediction proba. dist.')
        plt.xlabel('pred proba')
        plt.ylabel('statistics')
        plt.grid()

        return fig
        
    def show_plt(self):
        plt.show
        
    def save_fig(self, path : str):
        plt.savefig(path)

class TrendPlot:
    
    def __init__(self, figsize_x=8, figsize_y=6, is_time_series=True):

        plt.clf()
        plt.cla()
        self.fig, self.ax = plt.subplots()
        # plt.figure(figsize=(14, 4))
        self.fig.set_figheight(4)
        self.fig.set_figwidth(14)
        
        if is_time_series:
            dtFmt = mdates.DateFormatter('%b %y')
            plt.gca().xaxis.set_major_formatter(dtFmt)
            plt.xticks(rotation=45)

    def plot_trend(self, *args, label='unnammed trend'):
        self.ax.plot(*args, label=label)

    def plot_trend_with_error_bar(self, *args, **kwargs):
        self.ax.errorbar(*args, **kwargs)

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
            
    def plot_bar(self, *args, **kwargs):
        ax2 = self.ax.twinx()
        ax2.bar(*args, **kwargs)

    def save_fig(self, title=None, x_label='data list sequence', y_label='unnamed data', save_fig_path='./output_fig.pdf'):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.tick_params(axis="y", direction="inout")
        plt.tick_params(labelleft=True, labelright=True)
        plt.grid(axis='y', alpha=.7, linestyle=":")
        plt.tight_layout()
        plt.savefig(save_fig_path)


class RocCurve:

    def __init__(self, RocCurveDisplay):

        RocCurveDisplay.plot()
        plt.show()
        
        
        
if __name__ == '__main__':
    
    import pandas as pd
    x_date = ["2021-01-01 00:00", "2021-01-01 00:05", "2021-01-01 00:10", "2021-01-01 00:15"]
    x_date = pd.to_datetime(x_date)
    test_list = [3,2,6,4]
    trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
    trend_plot.plot_trend(x_date, test_list)
    trend_plot.save_fig(title='test plot', 
                        x_label='dummy time', 
                        y_label='dummy data',
                        save_fig_path='test.png')

