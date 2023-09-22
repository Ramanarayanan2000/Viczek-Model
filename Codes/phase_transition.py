#phase transition plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#system parameters
L = 5
N = 100
v = 0.03
rho = N/(L**2)
dt = 1

eta_list = np.linspace(0,5,15,endpoint = True)

order_param_store = []
for eta in eta_list:
    eta_op = []
    for w in range(5):

        #initial positions
        init_position = np.random.uniform(0,L,(N,2))
        
        #initial angles
        init_angle = np.random.uniform(-np.pi,np.pi,N)
        
        init_state = np.zeros((N,3))
        init_state[:,0] = init_position[:,0]
        init_state[:,1] = init_position[:,1]
        init_state[:,2] = init_angle
        
        
        
        #initial components of velocities
        init_velocity_components = np.zeros((N,2))
        init_velocity_components[:,0] = v*np.cos(init_angle)
        init_velocity_components[:,1] = v*np.sin(init_angle)
        
        #updating
        def state_updater(state):
            
            size = np.shape(state)
            dim1 = size[0]
            dim2 = size[1]
            
            angle_updates = np.zeros(dim1)
            for i in range(dim1):
                neighbour_angle = []
                for j in range(dim1):
                    
                    posn_i = np.array([state[i,0],state[i,1]])
                    posn_j = np.array([state[j,0],state[j,1]])
                    delta_posn = posn_j - posn_i
                    
                    if np.linalg.norm(delta_posn) < 1:
                        
                        neighbour_angle.append(state[j,2])
                
                sin_neighbour = np.sin(neighbour_angle)
                cos_neighbour = np.cos(neighbour_angle)
                
                avg_sin = np.mean(sin_neighbour)
                avg_cos = np.mean(cos_neighbour)
                
                avg_neighbour_angle = np.arctan(avg_sin/avg_cos)
                
                angle_updates[i] = avg_neighbour_angle
            
            angle_noise = np.random.uniform(-eta/2,eta/2,dim1)    
            angle_updates = angle_updates + angle_noise
        
            vx = v*np.cos(angle_updates)        
            vy = v*np.sin(angle_updates)
            
            x = state[:,0] + vx*dt
            #introducing PBC
            x = np.where(x > L,0,x)
            x = np.where(x < 0,L,x)
            
            y = state[:,1] + vy*dt
            #introducing PBC
            y = np.where(y > L,0,y)
            y = np.where(y < 0,L,y)
            
            
            new_state = np.zeros((dim1,3))
            new_state[:,0] = x
            new_state[:,1] = y
            new_state[:,2] = angle_updates
            
            return new_state
        
        def order_parameter(state):
            
            vx = v*np.cos(state[:,2])
            vy = v*np.sin(state[:,2])
            
            sum_velocity = np.array([sum(vx),sum(vy)])
            magnitude_sum_velocity = np.linalg.norm(sum_velocity)
            
            val = (1/(N*v))*magnitude_sum_velocity
            
            return val
        
        frames = 300
        input_state = init_state
        for s in range(frames):
            
            state_s = state_updater(input_state)
            input_state = state_s
            print(s,w,eta)
            
        eta_op.append(order_parameter(state_s))
        
    order_parameter = np.mean(eta_op)
    order_param_store.append(order_parameter)


#%%
#plotting

plt.figure(figsize=(11, 8.5))

plt.plot(eta_list,order_param_store,'o',markersize = 10,label = r'N = 100, $\rho$ = 4')    
plt.xlabel(r'$\eta$',fontsize = 25)
plt.xticks(fontsize = 20)
plt.ylabel( r'$v_a$',fontsize = 25)
plt.yticks(fontsize = 20)
plt.title(r'$v_a$ vs $\eta$',fontsize = 25)
plt.legend(prop = {'size':20})  
plt.savefig('phase_trans.png')
plt.show()
    
