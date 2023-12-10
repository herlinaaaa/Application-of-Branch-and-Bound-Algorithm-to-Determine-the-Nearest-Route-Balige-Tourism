from flask import Flask, render_template, request
import json
import heapq
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend explicitly
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load data from JSON file
with open('data_destinasi.json', 'r') as file:
    data = json.load(file)

destinations = data['destinations']
matrix = data['matrix']

def branch_and_bound(matrix, start, end):
    n = len(matrix)
    visited = [False] * n
    path = [start]
    cost = 0

    priority_queue = [(0, start, visited, path, cost)]

    while priority_queue:
        _, current, visited, path, cost = heapq.heappop(priority_queue)

        if current == end:
            return path, cost

        for neighbor in range(n):
            if not visited[neighbor] and matrix[current][neighbor] != float('inf'):
                new_visited = visited[:]
                new_visited[neighbor] = True
                new_path = path + [neighbor]
                new_cost = cost + matrix[current][neighbor]

                heapq.heappush(priority_queue, (new_cost, neighbor, new_visited, new_path, new_cost))

    return None, float('inf')

def find_nearest_destinations(matrix, destination, num_destinations=5, exclude_destinations=[]):
    n = len(matrix)
    destination_index = destinations.index(destination)

    distances = [(matrix[i][destination_index], destinations[i]) for i in range(n) if i != destination_index and destinations[i] not in exclude_destinations]
    distances.sort()

    return distances[:num_destinations]

def plot_route_and_nearest(start, end, path, nearest_destinations):
    G = nx.Graph()

    # Add edges for the fastest route
    route_edges = [(destinations[path[i]], destinations[path[i + 1]]) for i in range(len(path) - 1)]
    for edge in route_edges:
        G.add_edge(edge[0], edge[1], weight=matrix[destinations.index(edge[0])][destinations.index(edge[1])])

    # Add edges for the 5 nearest destinations from the destination
    for distance, dest in nearest_destinations:
        G.add_edge(end, dest, weight=distance)

    pos = nx.spring_layout(G)  # Set node positions with a spring layout

    node_colors = ['orange' if node == start else 'green' if node == end else 'skyblue' for node in G.nodes]

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title(f"Fastest Route from {start} to {end} and 5 Nearest Destinations")
    plt.savefig('static/graph_image.png')  # Save the graph image
    plt.close()  # Close the plot to prevent displaying it in the web app

# Modify the route to accept input from the user
@app.route('/', methods=['GET', 'POST'])
def shortest_path_web():
    start_point = "Kota Balige"
    end_point = "Bukit Pahoda"
    final_route = None

    if request.method == 'POST':
        # Get start_point and end_point from the form submission
        start_point = request.form['start_point']
        end_point = request.form['end_point']

        # Find the shortest path for the given start_point and end_point
        path, total_cost = branch_and_bound(matrix, destinations.index(start_point), destinations.index(end_point))

        if path:
            total_destinations_visited = len(path)
            total_distance = total_cost

            nearest_destinations = find_nearest_destinations(matrix, end_point, num_destinations=5, exclude_destinations=path)

            plot_route_and_nearest(start_point, end_point, path, nearest_destinations)

            final_route = {
                "path": [destinations[i] for i in path],
                "total_destinations_visited": total_destinations_visited,
                "total_distance": total_distance,
                "nearest_destinations": [(dest, distance) for distance, dest in nearest_destinations]
            }

    return render_template('shortest_path.html', final_route=final_route, start_point=start_point, end_point=end_point, destinations=destinations)

if __name__ == '__main__':
    app.run(debug=True)
