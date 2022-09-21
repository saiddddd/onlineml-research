import time


class TreeInspector:

    def __init__(self, model):
        self._model = model

    def update_model(self, model):
        self._model = model

    def draw_tree(self, output_fig_dir='', fig_file_name=''):
        tree = self._model

        g = tree.draw()
        g.render(output_fig_dir + fig_file_name+"_tree", format='png')


class HoeffdingEnsembleTreeInspector(TreeInspector):

    def __init__(self, model):

        super().__init__(model=model)

    def draw_tree(self, tree_index=0, output_fig_dir='', fig_file_name=''):
        trees = self._model.models
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

        trees = self._model.models
        return trees[tree_index].draw()

