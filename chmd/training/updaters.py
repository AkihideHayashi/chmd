"""Original Updaters."""
from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater


class GradUpdater(StandardUpdater):
    def __init__(self, iterator, optimizer, converter, device=None,
                 loss_func=None, grad_func=None,
                 loss_scale=None, auto_new_epoch=True):
        assert loss_func is None or grad_func is None
        super().__init__(iterator=iterator,
                         optimizer=optimizer,
                         converter=converter,
                         device=device,
                         loss_func=loss_func,
                         loss_scale=loss_scale,
                         auto_new_epoch=auto_new_epoch,
                         )
        self.grad_func = grad_func

    def update_core(self):
        if self.grad_func is None:
            super().update_core()
        else:
            iterator = self._iterators['main']
            batch = iterator.next()
            in_arrays = convert._call_converter(self.converter, batch, self.device)
            optimizer = self._optimizers['main']
            grad_func = self.grad_func
            if isinstance(in_arrays, tuple):
                grad_func(*in_arrays, target=optimizer.target)
            elif isinstance(in_arrays, dict):
                grad_func(**in_arrays, target=optimizer.target)
            else:
                grad_func(in_arrays, target=optimizer.target)
            optimizer.update()
            if self.auto_new_epoch and iterator.is_new_epoch:
                optimizer.new_epoch(auto=True)





