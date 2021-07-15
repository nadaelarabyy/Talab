import streamlit as st
import pandas as pd
import solver as solver
import matplotlib.pyplot as plt
import numpy as np

def read_ints(s):
    return [int(i) for i in s.split(' ')]
def read_value(input_file):
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

def draw(plt,rest_cust,del_cust):
    plt.scatter(np.array(restaurant_locations)[:,0],np.array(restaurant_locations)[:,1],marker='o',c='red',label="reaturants")
    plt.scatter(np.array(customer_locations)[:,0],np.array(customer_locations)[:,1],marker='x',c='blue',label="Customers")
    plt.scatter(np.array(delivery_locations)[:,0],np.array(delivery_locations)[:,1],marker='^',c='green',label="Couriers")
    plt.title("Input Locations")
    plt.legend()
    
    # tuple(delivery->restaurant)
    
    for i in range(custs):
        r=rest_cust[i]
        d=del_cust[i]
        if r == -1 or d==-1 :
            continue
        x=[customer_locations[i][0],restaurant_locations[r][0],delivery_locations[d][0]]
        y=[customer_locations[i][1],restaurant_locations[r][1],delivery_locations[d][1]]
        plt.plot(x,y,linestyle="dotted")

if __name__ == '__main__':
    st.title("Talab App")

    algo=st.sidebar.selectbox("Select an algorithm",("MIP","Greedy","Metaheuristic","DP"))
    # st.write(algo)
    # st.subheader("Dataset")
    data_file = st.sidebar.file_uploader("Upload File",type=['csv','in'])
    w=st.sidebar.number_input('Tune parameter',0,1000)
    if st.sidebar.button("Run"):
        if data_file is not None:
            file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
            # st.sidebar.write(file_details)
            if data_file.type == "csv":
                df = pd.read_csv(data_file)
                st.sidebar.dataframe(df)
            else:
                txt=[]
                for line in data_file:
                    txt.append(str(line,"utf-8"))
                custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders=read_value(txt)
                val = custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rests_dishes,clients_favs,customer_locations,restaurant_locations,delivery_locations,orders
                if algo == "MIP":
                    output = solver.MIP(val)
                    st.subheader("Objective Value")
                    st.success(output[0])
                    # st.write(plt.plot(rest_x,rest_y,'r',label="restaurant"))
                    rests = output[1][0]
                    courier = output[1][1]      
                    fig = plt.figure()
                    rest_cust=[-1]*custs
                    del_cust=[-1]*custs
                    for i,r in enumerate(rests):
                        if len(r)>0:
                            v=r.split(' ')
                            for v1 in v:
                                if len(v1)>0:
                                    rest_cust[int(v1)]=i
                    for i,d in enumerate(courier):
                        if len(d)>0:
                            del_cust[int(d)]=i
                    draw(plt,rest_cust,del_cust)
                    st.pyplot(fig=fig)      
                if algo == "DP":
                    output = solver.Dynamic(val)
                    st.subheader("Objective Value")
                    st.success(output[0])
                    output[1]=[output[2],output[1]]
                    rests = output[1][0]
                    courier = output[1][1]      
                    fig = plt.figure()
                    rest_cust=[-1]*custs
                    del_cust=[-1]*custs
                    for i,r in enumerate(rests):
                        if len(r)>0:
                            v=r.split(' ')
                            for v1 in v:
                                if len(v1)>0:
                                    rest_cust[int(v1)]=i
                    for i,d in enumerate(courier):
                        if len(d)>0:
                            del_cust[int(d)]=i
                    fig = plt.figure()
                    draw(plt,rest_cust,del_cust)
                    st.pyplot(fig=fig)
                if algo == "Greedy":
                    output = solver.Greedy(val)
                    st.subheader("Objective Value")
                    st.success(output[0])
                    fig = plt.figure()
                    draw(plt,output[1][0],output[1][1])
                    st.pyplot(fig=fig)
                if algo == "Metaheuristic":
                    output = solver.MetaHeuristic(val)
                    st.subheader("Objective Value")
                    st.success(output[0])
                    fig = plt.figure()
                    draw(plt,output[1][0],output[1][1])
                    st.pyplot(fig=fig)



            
