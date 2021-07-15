import sys
import os
from ortools.linear_solver import pywraplp
import math
import random
import pandas as pd
custs=0
rests=0
dgs=0
dishes =0
rl=0
vehs=0
w=0
vehicle_types=0
cpd=0
max_orders=0
rests_dishes=0
clients_favs=0
customer_locations=0
restaurant_locations=0
delivery_locations = 0
orders = 0
memo = {}
tabu = []
INF = 1000
def read_ints(s):
    return [int(i) for i in s.split(' ')]
def read_input(input_filepath):
    with open(input_filepath, 'r') as f:
        input_file = f.read().split('\n')
    custs,rests,dgs,dishes,rl,vehs,w = read_ints(input_file[0])
    vehicle_types = read_ints(input_file[1])
    cpd = read_ints(input_file[2])
    max_orders = read_ints(input_file[3])
    rests_dishes=[]
    for i in range(rests):
        lst = [0]*dishes
        c = read_ints(input_file[i+4])
        for n in c:
            lst[n] = 1
        rests_dishes.append(lst)
    clients_favs=[]
    for i in range(custs):
        lst = [0]*rests
        c = read_ints(input_file[i+4+rests])
        for n in c:
            lst[n] = 1
        clients_favs.append(lst)
    
    customer_locations =[]
    for i in range(custs):
        c =  read_ints(input_file[i + 4 + rests + custs])
        customer_locations.append(c)
    restaurant_locations = []
    for i in range(rests):
        c =  read_ints(input_file[i + 4 + rests + custs + custs])
        restaurant_locations.append(c)

    delivery_locations = []
    for i in range(dgs):
        c =  read_ints(input_file[i + 4 + rests + custs + custs + rests])
        delivery_locations.append(c)
    c = read_ints(input_file[-1])
    orders=[]
    for n in c:
        lst = [0]*dishes
        lst[n] = 1
        orders.append(lst)
    global INF
    INF = rl*max(cpd)*custs
    return custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders

def calcualte_dist(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))


def Dynamic(input_files_path,t):
    global memo,custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders
    custs=0
    rests=0
    dgs=0
    dishes =0
    rl=0
    vehs=0
    w=0
    vehicle_types=0
    cpd=0
    max_orders=0
    rests_dishes=0
    clients_favs=0
    customer_locations=0
    restaurant_locations=0
    delivery_locations = 0
    orders = 0
    memo = {}
    custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders =read_input(input_files_path+"/"+t+".in")
    return dp(0,0,[0]*rests)
def dp(current_customer,delivery,restaurants):
    string_ints = [str(int) for int in restaurants]
    restaurantsString = ''.join(string_ints)
    if (current_customer,delivery,restaurantsString) in memo:
        # print("using memo")
        return (memo[(current_customer,delivery,restaurantsString)])
    if delivery == (1<<dgs)-1:
        return [0,[""]*dgs,[""]*rests]

    if current_customer == custs:
        return [0,[""]*dgs,[""]*rests]


    max_solution = [0,[""]*dgs,[""]*rests]

    for i in range(0,dgs):
        if (delivery >> i ) & 1 == 0 :
            for j in range(0,rests):
                # print(i,j)
                current_distance = calcualte_dist(delivery_locations[i],restaurant_locations[j]) + calcualte_dist(restaurant_locations[j],customer_locations[current_customer])
                if restaurants[j] < max_orders[j] and rests_dishes[j][orders[current_customer].index(1)]==1 and clients_favs[current_customer][j]==1 and current_distance <rl:
                    new_restaurants = restaurants.copy()
                    new_restaurants[j]+=1
                    # current_distance = calcualte_dist(delivery_locations[i],restaurant_locations[j]) + calcualte_dist(restaurant_locations[j],customer_locations[current_customer])
                    solution = dp(current_customer+1,delivery|(1<<i),new_restaurants) 
                    copied = [0,[],[]]
                    copied[0] = solution[0] + INF
                    copied[0] -= w*current_distance*cpd[vehicle_types[i]]
                    copied[1] = solution[1].copy()
                    copied[2] = solution[2].copy()
                    copied[1][i] += " " + str(current_customer)
                    copied[2][j] += " " + str(current_customer)
                    if copied[0] > max_solution[0]:
                        max_solution = copied
                    
    if current_customer <custs:
        solution = dp(current_customer+1,delivery,restaurants) 
        if solution[0] > max_solution[0]:
            max_solution = solution

    memo[(current_customer,delivery,restaurantsString)] = max_solution
    return max_solution


def MIP(input_files_path,t):
    global custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders
    custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders =read_input(input_files_path+"/"+t+".in")
    # print("Orders",orders)
    # print("max orders",max_orders)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # xijkd
    xijkd=[]
    for d in range(dishes):
        xijk=[]
        for k in range(custs):
            xij=[]
            for j in range(rests):
                xi=[]
                for i in range(dgs):
                    xi.append(solver.IntVar(0,1,'x'+str(i)+str(j)+str(k)+str(d)))
                xij.append(xi)
            xijk.append(xij)
        xijkd.append(xijk)
    # print(xijkd)
    # maximum dishes per restaurants
    # j indicates restaurant
    for j in range(rests):
        sum_idk=0
        for i in range(dgs):
            for k in range(custs):
                for d in range(dishes):
                    sum_idk+=xijkd[d][k][j][i]
        solver.Add(sum_idk <= max_orders[j])
    # assign order to delivery
    for d in range(dishes):
        for j in range(rests):
            for k in range(custs):
                sum_i=0
                for i in range(dgs):
                    sum_i+=xijkd[d][k][j][i]
                solver.Add(sum_i <= rests_dishes[j][d]*clients_favs[k][j]*orders[k][d])
    # for the same delivery guys he cannot serve more than one customer.
    for i in range(dgs):
        sumjkd=0
        for j in range(rests):
            for k in range(custs):
                for d in range(dishes):
                    sumjkd+=xijkd[d][k][j][i]
        solver.Add(sumjkd <= 1)
    # distance constraint
    for d in range(dishes):
        for k in range(custs):
            for j in range(rests):
                for i in range(dgs):
                    current_distance = calcualte_dist(delivery_locations[i],restaurant_locations[j]) + calcualte_dist(restaurant_locations[j],customer_locations[k])
                    solver.Add(xijkd[d][k][j][i]*current_distance<=rl)
    # orders can only be prepared by only one restaurant and delivered by one guy.
    for d in range(dishes):
        for k in range(custs):
            sum_ij=0
            for j in range(rests):
                for i in range(dgs):
                    sum_ij+=xijkd[d][k][j][i]
            solver.Add(sum_ij<=1)
    obj_fn1=0
    for d in range(dishes):
        for k in range(custs):
            for j in range(rests):
                for i in range(dgs):
                    obj_fn1+=xijkd[d][k][j][i]
    obj_fn2=0
    for d in range(dishes):
        for k in range(custs):
            for j in range(rests):
                for i in range(dgs):
                    current_distance = calcualte_dist(delivery_locations[i],restaurant_locations[j]) + calcualte_dist(restaurant_locations[j],customer_locations[k])
                    obj_fn2+=xijkd[d][k][j][i]*current_distance*cpd[vehicle_types[i]]
    solver.Maximize(INF*obj_fn1-w*obj_fn2)


    output = []
    restaurants =['']*rests
    deliverys=['']*dgs
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL: 
        # print("objective value:",solver.Objective().Value())
        output.append(str(solver.Objective().Value()))
        for d in range(dishes):
            for k in range(custs):
                for j in range(rests):
                    for i in range(dgs):
                        v = xijkd[d][k][j][i].solution_value()
                        if v==1:
                            restaurants[j]+=" "+ str(k)
                            deliverys[i]+=" "+ str(k)
                            # print("delivery i:",i,"Located at",delivery_locations[i]," delivered order ",d," to customer ",k,"Located at",customer_locations[k]," from restaurant ",j,"Located at",restaurant_locations[j])
                            # print("Distance was:",calcualte_dist(delivery_locations[i],restaurant_locations[j]) + calcualte_dist(restaurant_locations[j],customer_locations[k]))
    else:
        print("no solution")
    output.append([restaurants,deliverys])
    return output



    
def Greedy(input_files_path,t):
    global custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders
    custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders =read_input(input_files_path+"/"+t+".in")
    matchedOrders = []
    matchings = []
    restaurants =[-1]*custs
    deliverys = [-1]*custs
    for clientId in range(len(orders)):
        possible_res = []
        for dishId in range(dishes):
            if(orders[clientId][dishId] == 1):
                # clientId = current client 
                # dishId = the dish that the client wants 
                clientsFavs = clients_favs[clientId]
                for res_index in range(len(clientsFavs)):
                    if(clientsFavs[res_index]==1):
                        if(rests_dishes[res_index][dishId]==1):
                            possible_res.append(res_index)
                matchedOrders.append(possible_res)
                matchings.append(len(possible_res))
                break

    # print(matchedOrders)
    # print(matchings)

    res_client_dist = []
    for i in range(len(matchedOrders)):
        temp_client_res=[]
        for j in range(len(matchedOrders[i])):
            temp_client_res.append(calcualte_dist(customer_locations[i], restaurant_locations[j]))
        res_client_dist.append(temp_client_res)
    
    rest_delivery_dist=[]
    for i in range(len(restaurant_locations)):
        delivery_res_temp=[]
        for j in range(len(delivery_locations)):
            delivery_res_temp.append(calcualte_dist(restaurant_locations[i],delivery_locations[j]))
        rest_delivery_dist.append(delivery_res_temp)

    max_iterations = max(matchings)

    isMatched = [0]*custs
    delivery_isMatched = [0]*dgs
    ## Match  the orders that has the minimum matching restaurants first 
    for i in range(max_iterations):
        ## for each clienf loop over all the restaurants that can deliver the client's order
        for j in range(len(matchedOrders)): ###Client
            Full_min_distance = math.inf
            Full_min_cost = math.inf
            restaurant_index = 0
            guy_min_index = -1
            min_res_guy_cost = math.inf
            min_res_guy_dist = math.inf
            guy_index = -1
            ## match the client with minimum number of matching restaurants first
            if matchings[j]<=(i+1):
                ## loop over all the possible restaurants to get the nearest possible one
                for k in range(len(matchedOrders[j])): ### Client_orders
                    temp_dist = res_client_dist[j][k]
                    ## loop over the delievry guys to get the nearest free one to the restaurent 
                    for g in range(dgs):
                        if(delivery_isMatched[g] == 0 ):
                            dist = rest_delivery_dist[matchedOrders[j][k]][g]
                            dist_cost = (rest_delivery_dist[matchedOrders[j][k]][g] + temp_dist ) * cpd[vehicle_types[g]]
                            if dist_cost < min_res_guy_cost:
                                min_res_guy_cost = dist_cost
                                min_res_guy_dist = dist
                                guy_min_index = g 
                    # min_distance = min(rest_delivery_dist[matchedOrders[j][k]])
                    # guy_min_index = rest_delivery_dist[matchedOrders[j][k]].index(min_distance)
                    
                    ##checking that the delievry guy is free 
                    ## that the total distance of the trip is less  than the maximum route length 
                    ## that the restaurent orders didn't reach the maximum that it can deliver
                    ## and that the order is not matched with another restaurant

                    full_distance = temp_dist + min_res_guy_dist
                    if min_res_guy_cost< Full_min_cost and full_distance < rl and isMatched[j]==0 and max_orders[matchedOrders[j][k]]>0 and delivery_isMatched[guy_min_index]==0:
                        Full_min_distance = full_distance
                        restaurant_index = matchedOrders[j][k]
                        guy_index = guy_min_index
            
            if(matchings[j]<=(i+1) and isMatched[j]==0 and guy_index>-1):
                isMatched[j] = 1 
                delivery_isMatched[guy_index] = 1
                max_orders[restaurant_index]-=1
                # print("Client ", j , " Order is matched with resturant " ,  restaurant_index," and delivery guy ", guy_index , " and total distance ", Full_min_distance , " and total cost ", min_res_guy_cost)
                restaurants[j] = restaurant_index
                deliverys[j] = guy_index
    return [getObjectiveValue([restaurants,deliverys]),[restaurants,deliverys]]

def getObjectiveValue(solution):
    # number of delivered orders*INF - transportation
    restaurants = solution[0]
    deliverys = solution[1]
    obj = 0
    for i in range(len(restaurants)):
        if solution[0][i] >-1:
            obj += INF - w*(calcualte_dist(delivery_locations[deliverys[i]],restaurant_locations[restaurants[i]])+calcualte_dist(restaurant_locations[restaurants[i]],customer_locations[i]))*cpd[vehicle_types[deliverys[i]]]
    return obj
def Neighbourhood(initial_solution):
    solutions = []
    #replacing every restaurant for clients
    
    for i in range(len(initial_solution[0])):
        for j in range(len(clients_favs[i])):
            if j!= initial_solution[0][i]:
                if clients_favs[i][j] == 1 and rests_dishes[j][orders[i].index(1)] == 1 and initial_solution[0].count(j) < max_orders[j]:
                    if initial_solution[0][i] != -1:
                        new_distance = calcualte_dist(delivery_locations[initial_solution[1][i]],restaurant_locations[j])+calcualte_dist(restaurant_locations[j],customer_locations[i])
                        if new_distance <= rl:
                            new_solution = [initial_solution[0].copy(),initial_solution[1].copy()]
                            new_solution[0][i] = j
                            if not(new_solution in solutions) and not(new_solution in tabu):
                                solutions.append(new_solution)
                    else:
                        for k in range(dgs):
                            if not k in initial_solution[1]:
                                new_distance = calcualte_dist(delivery_locations[k],restaurant_locations[j])+calcualte_dist(restaurant_locations[j],customer_locations[i])
                                if new_distance <= rl:
                                    new_solution = [initial_solution[0].copy(),initial_solution[1].copy()]
                                    new_solution[0][i] = j
                                    new_solution[1][i] = k
                                    if not(new_solution in solutions) and not(new_solution in tabu):
                                        solutions.append(new_solution)
                                        break



    for i in range(len(initial_solution[1])):
        for j in range(dgs):
            if j != initial_solution[1][i]:
                firstDistance = calcualte_dist(delivery_locations[j],restaurant_locations[initial_solution[0][i]])+calcualte_dist(restaurant_locations[initial_solution[0][i]],customer_locations[i])
                if firstDistance <= rl:
                    if j in initial_solution[1]:
                        index = initial_solution[1].index(j)
                        secondDistance = calcualte_dist(delivery_locations[initial_solution[1][i]],restaurant_locations[initial_solution[0][index]])+calcualte_dist(restaurant_locations[initial_solution[0][index]],customer_locations[index])
                        if secondDistance <= rl:
                            new_solution = [initial_solution[0].copy(),initial_solution[1].copy()]
                            new_solution[1][i] = j
                            new_solution[1][index] = initial_solution[1][i]
                            if not(new_solution in solutions) and not(new_solution in tabu):
                                solutions.append(new_solution)
                    else:
                        new_solution = [initial_solution[0].copy(),initial_solution[1].copy()]
                        new_solution[1][i] = j
                        if not(new_solution in solutions) and not(new_solution in tabu) :
                            solutions.append(new_solution) 

    return solutions

def getBestSolution(solutions):
    values = [0]*len(solutions)
    for i in range(len(solutions)):
        values[i] = getObjectiveValue(solutions[i])
    return solutions[values.index(max(values))]

def isBetter(new_best,solution):
    new_obj = getObjectiveValue(new_best)
    obj = getObjectiveValue(solution)
    return new_obj > obj
# solution = [[0,1,2],[0,1,2]] -- size is number of orders ex = 3
def MetaHeuristic(input_files_path,t):
    global tabu,custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders
    custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders =read_input(input_files_path+"/"+t+".in")
    tabu = []
    thetaMax = custs*rests*dgs
    step = thetaMax/100
    # thetaMax = 500
    # step = 50
    solution = [[-1]*custs,[-1]*custs]
    final_solution = solution.copy()
    for i in range(thetaMax):
        if i %step ==0 and len(tabu)>0:
            del tabu[0]
        n = Neighbourhood(solution)
        if len(n)>0:
            #get best non-tabu solution
            new_best = getBestSolution(n)
            # print("nada",new_best)
            #is there a better solution than initial one ?
            if isBetter(new_best,final_solution):
                final_solution = new_best
            tabu.append(new_best)
            solution = new_best

        if not solution in tabu:
            tabu.append(solution)

    return [getObjectiveValue(final_solution),final_solution]
    

def generateInput(custs,rests,dgs,dishes,rl,vehs,maxCPD,UB,t):
    w = random.randrange(0,2)


    vehicle_types = [0]*dgs
    for i in range(len(vehicle_types)):
        vehicle_types[i] = random.randrange(0,vehs)

        
    cpd = [0]*vehs
    for i in range(len(cpd)):
        cpd[i] = random.randrange(1,maxCPD)


    max_orders = [0]*rests
    for i in range(len(max_orders)):
        max_orders[i] = random.randrange(1,custs)


    rests_dishes =[]
    for i in range(rests):
        current_rest = []
        number = random.randrange(1,dishes)
        while len(current_rest) < number:
            for j in range(dishes):
                if len(current_rest) == number:
                    break
                if  random.randrange(0,2) == 1 and not j in current_rest:
                    current_rest.append(j)
        rests_dishes.append(current_rest)


    clients_favs = []
    for i in range(custs):
        current_client = []
        number = random.randrange(1,rests)
        while len(current_client) < number :
            for j in range(rests):
                if len(current_client) == number:
                    break
                if random.randrange(0,2) == 1 and not j in current_client:
                    current_client.append(j)
        clients_favs.append(current_client)


    customer_locations = []
    for i in range(custs):
        location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        while location in customer_locations:
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        customer_locations.append(location)


    restaurant_locations = []
    for i in range(rests):
        location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        while location in customer_locations or location in restaurant_locations:
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        restaurant_locations.append(location)


    delivery_locations = []
    for i in range(dgs):
        location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        while location in customer_locations or location in restaurant_locations or location in delivery_locations:
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
        delivery_locations.append(location)

    orders = []
    for i in range(custs):
        current_restaurant = random.choice(clients_favs[i])
        current_dish = random.choice(rests_dishes[current_restaurant])
        orders.append(current_dish)
    output = ""+str(custs)+" "+str(rests)+" "+str(dgs) +" "+str(dishes) +" "+str(rl)+" "+str(vehs)+" "+ str(w)+"\n"
    for v in vehicle_types:
        output+=str(v)+" "
    output = output[0:len(output)-1]
    output+="\n"

    for c in cpd:
        output += str(c) + " "
    output = output[0:len(output)-1]
    output+="\n"

    for m in max_orders:
        output+= str(m) + " "
    output = output[0:len(output)-1]
    output+="\n"

    for r in rests_dishes:
        for x in r:
            output+= str(x) + " "
        output = output[0:len(output)-1]
        output+="\n"

    for c in clients_favs:
        for x in c:
            output += str(x)+" "
        output = output[0:len(output)-1]
        output+="\n"
        
    for c in customer_locations:
        (x,y) = c
        output+=str(x)+" "+str(y)+"\n"
    for c in restaurant_locations:
        (x,y) = c
        output+=str(x)+" "+str(y)+"\n"
    for c in delivery_locations:
        (x,y) = c
        output+=str(x)+" "+str(y)+"\n"
    for o in orders:
        output+=str(o)+" "
    output = output[0:len(output)-1]
    

    with open(f"test_set/test_{t}.in","w")as myfile:
        myfile.write(output)
    return output
    
def generateInputExcel(custs,rests,dgs,dishes,rl,vehs,maxCPD,UB,t):
    cols = ['custs', 'rests', 'dishes','dgs', 'rl', 'vehs', 'w','vehicle types','cpd','max orders','rest dishes','client favs','customers locations','restaurant locations','delivery locations','orders']

    rows = []
    for i in range(t):
        w = random.randrange(0,2)


        vehicle_types = [0]*dgs
        for i in range(len(vehicle_types)):
            vehicle_types[i] = random.randrange(0,vehs)

            
        cpd = [0]*vehs
        for i in range(len(cpd)):
            cpd[i] = random.randrange(1,maxCPD)


        max_orders = [0]*rests
        for i in range(len(max_orders)):
            max_orders[i] = random.randrange(1,custs)


        rests_dishes =[]
        for i in range(rests):
            current_rest = []
            number = random.randrange(1,dishes)
            while len(current_rest) < number:
                for j in range(dishes):
                    if len(current_rest) == number:
                        break
                    if  random.randrange(0,2) == 1 and not j in current_rest:
                        current_rest.append(j)
            rests_dishes.append(current_rest)


        clients_favs = []
        for i in range(custs):
            current_client = []
            number = random.randrange(1,rests)
            while len(current_client) < number :
                for j in range(rests):
                    if len(current_client) == number:
                        break
                    if random.randrange(0,2) == 1 and not j in current_client:
                        current_client.append(j)
            clients_favs.append(current_client)


        customer_locations = []
        for i in range(custs):
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            while location in customer_locations:
                location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            customer_locations.append(location)


        restaurant_locations = []
        for i in range(rests):
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            while location in customer_locations or location in restaurant_locations:
                location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            restaurant_locations.append(location)


        delivery_locations = []
        for i in range(dgs):
            location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            while location in customer_locations or location in restaurant_locations or location in delivery_locations:
                location = (random.randrange(-UB,UB),random.randrange(-UB,UB))
            delivery_locations.append(location)

        orders = []
        for i in range(custs):
            current_restaurant = random.choice(clients_favs[i])
            current_dish = random.choice(rests_dishes[current_restaurant])
            orders.append(current_dish)
        vehicle_types="#".join(map(str,vehicle_types))
        cpd="#".join(map(str,cpd))
        max_orders = "#".join(map(str,max_orders))
        for i,rd in enumerate(rests_dishes):
            rests_dishes[i]=",".join(map(str,rd))
        rests_dishes = "#".join(map(str,rests_dishes))
        for i,cv in enumerate(clients_favs):
            clients_favs[i]=",".join(map(str,cv))
        clients_favs = "#".join(map(str,clients_favs))
        for i,cl in enumerate(customer_locations):
            customer_locations[i]=str(cl[0])+","+str(cl[1])
        customer_locations = "#".join(customer_locations)
        for i,rll in enumerate(restaurant_locations):
            restaurant_locations[i]=str(rll[0])+","+str(rll[1])
        restaurant_locations = "#".join(restaurant_locations)
        for i,dl in enumerate(delivery_locations):
            delivery_locations[i]=str(dl[0])+","+str(dl[1])
        delivery_locations = "#".join(delivery_locations)
        orders="#".join(map(str,orders))
        rows = rows + [[custs,rests,dishes,dgs,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders]]
    summary = pd.DataFrame(rows, columns=cols)
    summary.to_csv("big_sets/summary.csv", index=False)
    
    # output = ""+str(custs)+" "+str(rests)+" "+str(dgs) +" "+str(dishes) +" "+str(rl)+" "+str(vehs)+" "+ str(w)+"\n"
    # for v in vehicle_types:
    #     output+=str(v)+" "
    # output = output[0:len(output)-1]
    # output+="\n"

    # for c in cpd:
    #     output += str(c) + " "
    # output = output[0:len(output)-1]
    # output+="\n"

    # for m in max_orders:
    #     output+= str(m) + " "
    # output = output[0:len(output)-1]
    # output+="\n"

    # for r in rests_dishes:
    #     for x in r:
    #         output+= str(x) + " "
    #     output = output[0:len(output)-1]
    #     output+="\n"

    # for c in clients_favs:
    #     for x in c:
    #         output += str(x)+" "
    #     output = output[0:len(output)-1]
    #     output+="\n"
        
    # for c in customer_locations:
    #     (x,y) = c
    #     output+=str(x)+" "+str(y)+"\n"
    # for c in restaurant_locations:
    #     (x,y) = c
    #     output+=str(x)+" "+str(y)+"\n"
    # for c in delivery_locations:
    #     (x,y) = c
    #     output+=str(x)+" "+str(y)+"\n"
    # for o in orders:
    #     output+=str(o)+" "
    # output = output[0:len(output)-1]
    

    # with open(f"test_set/test_{t}.in","w")as myfile:
    #     myfile.write(output)
    # return output
    
if __name__ == '__main__':
    input_files_path = sys.argv[1] # testset input
    # generateInput(custs,rests,dgs,dishes,rl,vehs,maxCPD,UB,t)
    # for i in range(2,11):
    print(generateInputExcel(10,10,10,6,200,5,20,200,20))
    for test_case in sorted(os.listdir(f'{input_files_path}'), key=lambda x: int(x[5:-3])):
        t = test_case[:-3]
        print("Solving",t)
        # print(MIP(input_files_path,t))
        # print(Dynamic(input_files_path,t))
        # print(Greedy(input_files_path,t))
        # print(MetaHeuristic(input_files_path,t))
            

