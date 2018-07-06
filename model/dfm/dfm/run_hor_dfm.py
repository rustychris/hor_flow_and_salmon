"""
Not used right now, but could be used for invoking all or part of
the model setup.
"""

import six
import hor_dfm

six.moves.reload_module(hor_dfm)

model=hor_dfm.model

model.write()

model.partition()

model.run_model()




