import heapq

# Define the graph from the image
graph = {
    1: {2: 4, 3: 3},
    2: {1: 4, 4: 1, 5: 1.5, 3:2},
    3: {1: 3, 2: 2, 8: 8, 6:3},
    4: {2: 1, 7: 5, 5:3},
    5: {2: 1.5, 4: 3, 8: 4},
    6: {3: 3, 8: 1},
    7: {4: 5, 8: 10},
    8: {5: 4, 6: 1, 7: 10, 3:8},
}

# Dijkstra's Algorithm to compute shortest paths from a source node to all other nodes
def dijkstra(graph, start):
    # Priority queue to hold all nodes and the cost to reach them
    pq = [(0, start, [])]
    # A set to hold all visited nodes
    visited = set()
    # Dict to hold the shortest path to a node as well as the path taken (via nodes)
    shortest_paths = {start: (0, [])}

    while pq:
        (cost, node, path) = heapq.heappop(pq)
        # Skip nodes already visited
        if node in visited:
            continue
        # Add node to visited
        visited.add(node)
        # Update path
        path = path + [node]
        # Update shortest paths dict if we found a shorter path to the node
        shortest_paths[node] = (cost, path)
        
        # Look at neighbors of the current node
        for neighbor in graph[node]:
            if neighbor not in visited:
                # Total cost of path to the neighbor
                total_cost = cost + graph[node][neighbor]
                # Push neighbor to the priority queue
                heapq.heappush(pq, (total_cost, neighbor, path))
    
    return shortest_paths

# Running the algorithm from node 1
shortest_paths = dijkstra(graph, 1)

# Construct the table following the image's format (Newly added node, NextHop, Total cost)
routing_table = []

# Assuming 'S' is a set that will contain the nodes in the order they are added
S = set()
for node, (cost, path) in sorted(shortest_paths.items(), key=lambda x: x[1][0]):
    S.add(node)
    next_hop = path[1] if len(path) > 1 else '-'  # Next hop is the second node in the path, if available
    routing_table.append((node, next_hop, cost, S.copy()))

print(routing_table)