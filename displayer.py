bot_template = '''
<div class="chat-message bot" style="padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; justify-content: space-around; align-items: center; background-color: #FFFFFF; color: #000; border: 1px solid #228A5E;">
    <div class="avatar">
        <img src="https://t4.ftcdn.net/jpg/02/20/70/95/360_F_220709518_gEZaavNnrkwpcfZV2sIPOJfWEV9b8n4f.jpg" 
        style="max-height: 4.5rem; max-width: 4.5rem; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message" style="text-align: left; width: 80%; padding: 0 1.5rem; color: #000;">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user" style="padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; justify-content: space-around; align-items: center; background-color: #228A5E; color: #000; flex-direction: row-reverse;">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/47/47774.png"
        style="max-height: 4.5rem; max-width: 4.5rem; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message" style="text-align: end; width: 80%; padding: 0 1.5rem; color: #000;">{{MSG}}</div>
</div>
'''
