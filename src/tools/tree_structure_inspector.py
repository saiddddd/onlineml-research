import time


class HoeffdingEnsembleTreeInspector:

    def __init__(self, model):

        self.__model = model

    def draw_tree(self, tree_index, output_fig_dir):
        trees = self.__model.models
        timestamp = time.time()
        if tree_index is None:
            for i in range(len(trees)):
                g = trees[i].draw()
                g.render(output_fig_dir + "tree_inspect/AdaRF_tree{}_Structure_after_training_{}".format(i, timestamp), format='png')
        else:
            g = trees[tree_index].draw()
            g.render(output_fig_dir + "tree_inspect/AdaRF_tree{}_Structure_after_training_{}".format(tree_index, timestamp), format='png')