# Aston Hack 11

# Our project 
During the Astroworld Festival in 2021, a fatal crowd crush killed eight people during the tradegy and 2 more in the following days. Initial plans for the event contained contingencies for a wide range of scenarios, however there were no plans for crowd surge or mosh pit safety. 

In 2022, a crowd surge during Halloween celebrations in Seoul resulted in 159 deaths. The police later stated that they did not have a crowd control plan in place. 

These tradegies could have been avoided, if authorities and event organisers were given the necessary information to make their event as safe as possible. This is the motivation for our project Crowd Flow. We model high attendance events such as concerts and provide organisers a predictive framework to safeguard their communities.

# How we model large scale events
Our project uses physics to model thousands of event attendees. We mimic human movement by building our solution around the Social Force Model with Verlet Integration. This allows us to mimic the fluid-like dynamics of human crowds. To scale to multiple thousand attendees, we had to use mutiple optimisations such as utilising Numba's Just-In-Time compilation on our critical physics loops or implementing Spatial Partitioning via Cell Lists to reduce time complexity when calculating forces acting on attendees. 

Crowds at concerts don't just congregate at one location, they also react to music differently depending on the songs RMS Energy, Onset Strength and Tempo. Our project can analyse waveforms of music and then present this in the model in real time. During hype moments the crowd will act more erratically, while in calmer songs the crowd is more subdued and attendees will wander around more. This gives an event organiser a key insight into how and when dangerous situations can form. Furthermore, attendee wellbeing is effectively visualised for organisers with each attendees' physical pressure colored according to how cramped they are and a graph showing in real time the average pressure being felt by attendees. 

# How do we actually solve the problem using the model
Our model is able accurately visualise and simulate crowds at a concert, with key insights such as attendee pressure easily identifiable. So the question becomes how do organisers reduce this pressure to help protect their communities? Identifying the problem is easy, solving it is hard. Before this model, organisers guess where to place barriers to reduce pressure. We don't guess. Our Monte Carlo Search Engine rapidly simulates hundreds of barrier configurations, applying strategic patterns funnels and wave-breakers to minimize the average pressure felt by a crowd. An organiser is now able to protect their community using a solved equation. 
