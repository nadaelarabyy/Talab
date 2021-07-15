import enum
from random import random
from pandas.core.indexes.base import Index
import streamlit as st
import pandas as pd
import solver as solver
import matplotlib.pyplot as plt
import numpy as np
import random
import SessionState 

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
    plt.scatter(np.array(restaurant_locations)[:,0],np.array(restaurant_locations)[:,1],marker='o',c='red',label="Restaurants")
    plt.scatter(np.array(client_locations)[:,0],np.array(client_locations)[:,1],marker='x',c='blue',label="Customers")
    plt.scatter(np.array(delivery_locations)[:,0],np.array(delivery_locations)[:,1],marker='^',c='green',label="Couriers")
    plt.title("Map Visualization")
    plt.legend()
    
    # tuple(delivery->restaurant)
    
    for i in range(custs):
        r=rest_cust[i]
        d=del_cust[i]
        if r == -1 or d==-1 :
            continue
        x=[client_locations[i][0],restaurant_locations[r][0],delivery_locations[d][0]]
        y=[client_locations[i][1],restaurant_locations[r][1],delivery_locations[d][1]]
        types = ['-', '--', '-.', ':']
        plt.plot(x,y,linestyle=types[random.randrange(0,len(types))])

def run (val,algo):
    if algo == "MIP":
        output = solver.MIP(val)
        st.subheader("Total Objective Value")
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
        st.subheader("Number of delivered orders out of "+str(custs))
        st.success(len(del_cust)-del_cust.count(-1))

        st.subheader("Total transportation cost for the used vehicles "+str(len(del_cust)-del_cust.count(-1)))
        st.error(solver.getCost([rest_cust,del_cust]))

        st.subheader("Output solution")
        data = {'Client Number': [i for i in range(custs)], 'Restaurants': rest_cust,'Delivery guys': del_cust}  
        df = pd.DataFrame(data)

        st.table(df)


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
        st.subheader("Number of delivered orders out of "+str(custs))
        st.success(len(del_cust)-del_cust.count(-1))

        st.subheader("Total transportation cost for the used vehicles "+str(len(del_cust)-del_cust.count(-1)))
        st.error(solver.getCost([rest_cust,del_cust]))

        st.subheader("Output solution")
        data = {'Client Number': [i for i in range(custs)], 'Restaurants': rest_cust,'Delivery guys': del_cust}  
        df = pd.DataFrame(data)
        st.table(df)

        draw(plt,rest_cust,del_cust)
        st.pyplot(fig=fig)
    if algo == "Greedy":
        output = solver.Greedy(val)
        st.subheader("Objective Value")
        st.success(output[0])
        fig = plt.figure()
        st.subheader("Number of delivered orders out of "+str(custs))
        st.success(len(output[1][1])-output[1][1].count(-1))
        st.subheader("Total transportation cost for the used vehicles: "+str(dgs - output[1][1].count(-1)))
        st.error(solver.getCost(output[1]))

        st.subheader("Output solution")
        data = {'Client Number': [i for i in range(custs)], 'Restaurants': output[1][0],'Delivery guys': output[1][1]}  
        df = pd.DataFrame(data)
        st.table(df)

        draw(plt,output[1][0],output[1][1])
        st.pyplot(fig=fig)
    if algo == "Metaheuristic":
        output = solver.MetaHeuristic(val)
        st.subheader("Objective Value")
        st.success(output[0])
        fig = plt.figure()
        st.subheader("Number of delivered orders out of "+str(custs))
        st.success(len(output[1][1])-output[1][1].count(-1))
        st.subheader("Total transportation cost for the used vehicles: "+str(dgs - output[1][1].count(-1)))
        st.error(solver.getCost(output[1]))

        st.subheader("Output solution")
        data = {'Client Number': [i for i in range(custs)], 'Restaurants': output[1][0],'Delivery guys': output[1][1]}  
        df = pd.DataFrame(data)

        st.table(df)

        draw(plt,output[1][0],output[1][1])
        st.pyplot(fig=fig)

if __name__ == '__main__':
    st.title("Talab App")
    session = SessionState.get(run_id=0)

    if st.button("Reset"):
        session.run_id += 1


    global client_locations,restaurant_locations,delivery_locations
    vehicle_types = []
    client_locations = []
    restaurant_locations = []
    delivery_locations = []
    client_favs = []
    rest_dishes = []
    orders=[]
    cpd = []
    w = 0
    max_orders=[]
    choice = st.sidebar.radio("Choose",["Input field","BULK CSV"],key = str(session.run_id)+"radio")
    algo=st.sidebar.selectbox("Select an algorithm",("MIP","Greedy","Metaheuristic","DP"),key = str(session.run_id)+'"selectbox')
    if choice == "Input field":
        col1, col2,col3 = st.beta_columns([1,1,1])
        custs = col1.number_input('Number of Customers',1,1000,key = str(session.run_id)+"custs")
        rests = col2.number_input('Number of Restaurants',1,1000,key = str(session.run_id)+"rests")
        dgs = col3.number_input('Number of Delivery guys',1,1000,key = str(session.run_id)+"dgs")
        rl = col1.number_input('Maximum Route Length',1,1000,key = str(session.run_id)+"rl")
        vehs = col2.number_input('Number of Vehicles',1,1000,key = str(session.run_id)+"vehs")
        dishes = col3.number_input('Number of Dishes',1,1000,key = str(session.run_id)+"dishes")
        st.markdown("""---""")
        st.subheader("""Cost per distance for each vehicle type""")
        cola,colb,colc,cold = st.beta_columns([1,1,1,1])
        for c in range(vehs):
            if c%4 == 0:
                cpd.append(cola.number_input("CPD for vehicle type: "+str(c),1,1000,key = str(session.run_id)+"Vehicle"+str(c)))
            if c%4 == 1:
                cpd.append(colb.number_input("CPD for vehicle type: "+str(c),1,1000,key = str(session.run_id)+"Vehicle"+str(c)))
            if c%4 == 2:
                cpd.append(colc.number_input("CPD for vehicle type: "+str(c),1,1000,key = str(session.run_id)+"Vehicle"+str(c)))
            if c%4 == 3:
                cpd.append(cold.number_input("CPD for vehicle type: "+str(c),1,1000,key = str(session.run_id)+"Vehicle"+str(c)))
        st.markdown("""---""")
        st.subheader("""Customer Locations""")
        col4, col5,col6 = st.beta_columns([1,1,1])
        for c in range(custs):
            x = col4.number_input("Client Number: "+str(c)+" X",-1000,1000,key = str(session.run_id)+"Clientx"+str(c))
            y = col5.number_input("Client Number: "+str(c)+" Y",-1000,1000,key = str(session.run_id)+"Clienty"+str(c))
            o = col6.selectbox("Client Number: "+str(c)+" Order",[i for i in range(dishes)],key = str(session.run_id)+"orders"+str(c))
            favs = st.multiselect("Client Number: "+str(c)+" Favourite restaurants", [i for i in range(rests)],key = str(session.run_id)+"Favourite rests"+str(c))
            fulllist = [0 for _ in range(dishes)]
            fulllist[o] = 1
            client_locations.append((x,y))
            orders.append(fulllist)
            fulllist = [0 for _ in range(rests)]
            for x in favs:
                fulllist[x] = 1
            client_favs.append(fulllist)
            col4, col5,col6 = st.beta_columns([1,1,1])
        st.markdown("""---""")
        st.subheader("""Restaurants Locations""")
        col7, col8,col9 = st.beta_columns([1,1,1])
        for c in range(rests):
            x = col7.number_input("Restaurant Number: "+str(c)+" X",-1000,1000,key = str(session.run_id)+"rx"+str(c))
            y = col8.number_input("Restaurant Number: "+str(c)+" Y",-1000,1000,key = str(session.run_id)+"ry"+str(c))
            max_orders.append(col9.number_input("Restaurant Number: "+str(c)+" Maximum dishes",1,dishes,key = str(session.run_id)+"maxdishes"+str(c)))
            av = st.multiselect("Restaurant Number: "+str(c)+" available dishes", [i for i in range(dishes)],key = str(session.run_id)+"available"+str(c))
            restaurant_locations.append((x,y))
            fulllist = [0 for _ in range(dishes)]
            for x in av:
                fulllist[x] = 1
            rest_dishes.append(fulllist)

            col7, col8,col9 = st.beta_columns([1,1,1])
        st.markdown("""---""")
        st.subheader("""Delivery guys Locations""")
        col10, col11,col12 = st.beta_columns([1,1,1])
        for c in range(dgs):
            x = col10.number_input("Delivery guy Number: "+str(c)+" X",-1000,1000,key = str(session.run_id)+"dx"+str(c))
            y = col11.number_input("Delivery guy Number: "+str(c)+" Y",-1000,1000,key = str(session.run_id)+"dy"+str(c))
            delivery_locations.append((x,y))
            vehicle_types.append(col12.selectbox("Delivery guy vehicle type: "+str(c)+" ",[i for i in range(vehs)],key = str(session.run_id)+"vehstype"+str(c)))
        if st.checkbox("Check if  you want to minimize transportation costs",key = str(session.run_id)+"w"):
            w = 1
        if st.sidebar.button("Run"):
            val = custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rest_dishes,client_favs,client_locations,restaurant_locations,delivery_locations,orders
            run(val,algo)

    else:
    
        # st.write(algo)
        # st.subheader("Dataset")
        data_file = st.file_uploader("Upload File",type=['csv'])
        if st.sidebar.button("Run"):
            if data_file is not None:
                file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                df = pd.read_csv(data_file)
                for row_number in range(len(df)):
                    custs = df.iloc[row_number,0]
                    rests = df.iloc[row_number,1]
                    dgs = df.iloc[row_number,2]
                    dishes = df.iloc[row_number,3]
                    rl = df.iloc[row_number,4]
                    vehs = df.iloc[row_number,5]
                    w = df.iloc[row_number,6]


                    vehicle_types = df.iloc[row_number,7]
                    cpd = df.iloc[row_number,8]
                    max_orders = df.iloc[row_number,9]
                    rest_dishes = df.iloc[row_number,10]
                    client_favs = df.iloc[row_number,11]
                    client_locations = df.iloc[row_number,12]
                    restaurant_locations = df.iloc[row_number,13]
                    delivery_locations = df.iloc[row_number,14]
                    orders = df.iloc[row_number,15]

                    vehicle_types = vehicle_types.split("#")
                    vehicle_types = [int(i) for i in vehicle_types]

                    cpd = cpd.split("#")
                    cpd = [int(i) for i in cpd]

                    max_orders = max_orders.split("#")
                    max_orders = [int(i) for i in max_orders]

                    
                    rest_dishes = rest_dishes.split("#")
                    for i in range(len(rest_dishes)):
                        av = rest_dishes[i].split(',')
                        rest_dishes[i] = [int (x) for x in av]
                    fulllist = []
                    for i in range(rests):
                        currentlist = [0]*dishes
                        for x in rest_dishes[i]:
                            currentlist[x] = 1
                        fulllist.append(currentlist)
                    rest_dishes = fulllist
        ###########################################################
                    client_favs = client_favs.split("#")
                    for i in range(len(client_favs)):
                        cv = client_favs[i].split(',')
                        client_favs[i] = [int (x) for x in cv]
                    fulllist = []
                    for i in range(custs):
                        currentlist = [0]*rests
                        for x in client_favs[i]:
                            currentlist[x] = 1
                        fulllist.append(currentlist)
                    client_favs = fulllist
                    ####################################
                    client_locations = client_locations.split("#")
                    for i in range(len(client_locations)):
                        l = client_locations[i].split(",")
                        client_locations[i] = (int(l[0]),int(l[1]))

                    restaurant_locations = restaurant_locations.split("#")
                    for i in range(len(restaurant_locations)):
                        l = restaurant_locations[i].split(",")
                        restaurant_locations[i] = (int(l[0]),int(l[1]))

                    delivery_locations = delivery_locations.split("#")
                    for i in range(len(delivery_locations)):
                        l = delivery_locations[i].split(",")
                        delivery_locations[i] = (int(l[0]),int(l[1]))


                    ######################################
                    orders = orders.split("#")
                    for i in range(len(orders)):
                        dv = orders[i].split(',')
                        orders[i] = [int (x) for x in dv]
                    fulllist = []
                    for i in range(custs):
                        currentlist = [0]*dishes
                        for x in orders[i]:
                            currentlist[x] = 1
                        fulllist.append(currentlist)
                    orders = fulllist
                    val = custs,rests,dgs,dishes,rl,vehs,w,vehicle_types,cpd,max_orders,rest_dishes,client_favs,client_locations,restaurant_locations,delivery_locations,orders
                    run(val,algo)






            
