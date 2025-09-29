from pgmpy.estimators import HillClimbSearch, BIC
import subprocess
from pgmpy.models.DiscreteBayesianNetwork import DiscreteBayesianNetwork as BayesianModel
import networkx as nx

def learn_graph(data, dominant_data_type):

    if dominant_data_type == "discrete":
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-d')
        #edge_list = model.edges()
        #column_names = list(model.get_markov_blanket('target'))
        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        column_names = output.split()
    else:
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-g')
        #edge_list = model.edges()
        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        column_names = output.split()

    return column_names

def learn_graph_exc_desc(data, dominant_data_type):

    if dominant_data_type == "discrete":
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-d')
        #edge_list = model.edges()
        #bayes_model = BayesianModel(model.edges())  # convert DAG to BayesianModel
        #descendants = list(nx.descendants(bayes_model, "gender"))
        result = subprocess.run(
            ['Rscript', 'bnlearn_desc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        descendants = output.split()
    else:
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-g')
        #edge_list = model.edges()
        result = subprocess.run(
            ['Rscript', 'bnlearn_desc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        descendants = output.split()

    return descendants

def learn_graph_exc_desc_mb(data, dominant_data_type):

    if dominant_data_type == "discrete":
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-d')
        #edge_list = model.edges()
        #column_names = list(model.get_markov_blanket('target'))
        #bayes_model = BayesianModel(model.edges())  # convert DAG to BayesianModel
        #descendants = list(nx.descendants(bayes_model, "gender"))
        result = subprocess.run(
            ['Rscript', 'bnlearn_desc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        descendants = output.split()

        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        column_names = output.split()
    else:
        #est = HillClimbSearch(data)
        #model = est.estimate(scoring_method='bic-g')
        #edge_list = model.edges()
        result = subprocess.run(
            ['Rscript', 'bnlearn_desc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        descendants = output.split()

        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

        # Capture the output of the R script
        output = result.stdout.strip()  # Remove any trailing newline or spaces

        # Convert the space-separated output into a Python list
        column_names = output.split()

    return column_names, descendants
