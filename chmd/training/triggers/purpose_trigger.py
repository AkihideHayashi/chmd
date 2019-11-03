import operator
import warnings
from chainer.training import util
from chainer import reporter


class PurposeTrigger(object):
    def __init__(self, tol, check_trigger=(1, 'epoch'), monitor='main/loss',
                 max_trigger=(100, 'epoch')):
        self.count = 0
        self._max_trigger = util.get_trigger(max_trigger)
        self._interval_trigger = util.get_trigger(check_trigger)
        self.monitor = monitor
        self.tol = tol
        self._init_summary()
        self._compare = operator.lt

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary
        if self.monitor in observation:
            summary.add({self.monitor: observation[self.monitor]})
        if self._max_trigger(trainer):
            return True
        if not self._interval_trigger(trainer):
            return False
        if self.monitor not in observation.keys():
            warnings.warn('{} is not in observation'.format(self.monitor))
            return False
        stat = self._summary.compute_mean()
        current_val = stat[self.monitor]
        self._init_summary()
        if self._compare(current_val, self.tol):
            return True
        self.count += 1
        return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def get_training_length(self):
        return self._max_trigger.get_training_length()
