
def stats(dataset):
    stats = { 
        degree : 0,
        graphlet4 : 0,
        graphlet5 : 0 
    }

    for i, graph in enumerate(dataset):
        for degree in graph.degrees:
            stats['degree'] = stats['degree'] + 1

        for node in graph.graphlet4:
            for n, motif in enumerate(node):
                if motif > 0:
                    stats['graphlet4'] = stats['graphlet4'] + 1
                    
        for node in graph.graphlet5:
            for n, motif in enumerate(node):
                if motif > 0:
                    stats['graphlet5'] = stats['graphlet5'] + 1

    return stats
