import LAXOR_Sim.Config as config
import numpy as np

class performance:
    def __init__(self):
        self.a = 0

    def get_performace(self, total_cycle, total_cycle_PE):
        performance = (total_cycle-total_cycle_PE) * config.CLOCK_PERIOD_OTHERS + total_cycle_PE * config.CLOCK_PERIOD_PE


        return performance
