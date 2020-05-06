import autofit as af


class PipelineDataset(af.Pipeline):
    def run(self, dataset, mask, info=None):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results, mask=mask, info=info)

        return self.run_function(runner)
