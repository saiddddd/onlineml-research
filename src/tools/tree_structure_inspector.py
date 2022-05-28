import time


class HoeffdingEnsembleTreeInspector:

    def __init__(self, model):

        self.__model = model

    def draw_tree(self, tree_index, output_fig_dir, fig_file_name=''):
        trees = self.__model.models
        timestamp = time.time()

        if len(fig_file_name) == 0:
            fig_file_name = "AdaRF_inspect_{}".format(timestamp)

        if tree_index is None:
            for i in range(len(trees)):
                # fig_file_name = fig_file_name+"_tree_{}".format(i)
                g = trees[i].draw()
                g.render(output_fig_dir + fig_file_name+"_tree_{}".format(i), format='png')
        else:
            g = trees[tree_index].draw()
            g.render(output_fig_dir + fig_file_name, format='png')

    def get_tree_g(self, tree_index: int):

        trees = self.__model.models
        return trees[tree_index].draw()

