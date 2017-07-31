import pymysql
import json

## get auth data
with open('auth/mysql_connection_data.json') as data_file:    
    auth = json.load(data_file)

def derive(review):
    ## print review
    #print(review);
    
    ## open mysql connection
    conn = pymysql.connect(user=auth["user"], passwd=auth["password"], db=auth["database"], host='localhost')
    cur = conn.cursor(pymysql.cursors.DictCursor)
    
    ## get internal db id from ext db id
    ext_id = review["database_id"];
    print("Running " + ext_id);
    sql = "SELECT ReviewID FROM reviews WHERE ExtID ='%s'";
    cur.execute(sql % ext_id)
    internal_id = None;
    for row in cur:
        ##print(row)
        internal_id = row["ReviewID"];
    if(internal_id is None):
        print ("   `-> review not found (ext_id = " + ext_id + ")");
        return False;
    
    ## grab all scores associated with internal db id 
    sql = "SELECT Aspect, Value FROM scores WHERE ReviewID ='%s'";
    cur.execute(sql % internal_id)
    user_ratings = dict();
    for row in cur:
        user_ratings[row["Aspect"]] = row["Value"];
    ##print(user_ratings)
    
    cur.close()
    conn.close()
    
    ## sum the scores for each aspect from review data extraction 
    approved_aspects = ["location", "service", "price"];
    extracted_ratings = dict();
    for sentence in review["data"]:
        ##print("new sentense : ")
        ##print(sentence);
        for sentiment, sentiment_value in sentence["sentiment"].items():
            if(sentiment == "compound"): continue;
            for aspect in sentence["aspect"]:
                if(aspect[0] not in approved_aspects): continue;
                this_key = aspect[0] + "_" + sentiment;
                if this_key not in extracted_ratings: extracted_ratings[this_key] = 0.0;
                extracted_ratings[this_key] += float(aspect[1])*float(sentiment_value);
    ##print(extracted_ratings);
    
    ## convert user ratings to be keyed like extracted ratings
    mapping = dict({
        "Location" : "location",
        "Service" : "service",
        "Value" : "price",
        "Cleanliness" : None,
        "Rooms" : None,
        "Sleep Quality" : None,
        "Business service (e.g., internet access)" : None,
    })
    mapped_user_ratings = dict();
    for key, value in user_ratings.items():
        new_key = mapping[key];
        if(new_key is None): continue;
        mapped_user_ratings[new_key] = int(value);
    ## ensure all aspects are defiend
    for aspect in approved_aspects:
        if(aspect not in mapped_user_ratings): mapped_user_ratings[aspect] = -1;
    ##print(mapped_user_ratings);
        
        
    #############
    ## Note, the following data is defined explicitly for maintainability and readability. The extra time is worth it.
    #############
    
    ## define data labels: ext_review_id, label_location, label_service, label_price, f_1,1 through f_3,3 
    header_array = ["ext_review_id", "label_location", "label_service", "label_price", "location_neg", "location_neu", "location_pos", "service_neg", "service_neu", "service_pos", "price_neg", "price_neu", "price_pos"];
    
    ## define this derived data
    values = [ext_id, mapped_user_ratings["location"], mapped_user_ratings["service"], mapped_user_ratings["price"], extracted_ratings["location_neg"], extracted_ratings["location_neu"], extracted_ratings["location_pos"], extracted_ratings["service_neg"], extracted_ratings["service_neu"], extracted_ratings["service_pos"], extracted_ratings["price_neg"], extracted_ratings["price_neu"], extracted_ratings["price_pos"]];
    
    ## to string
    csv_header_string = ", ".join(header_array);
    csv_data_string = ", ".join([str(value) for value in values]);
    
    result = ([csv_header_string, csv_data_string]);
    return result;
    
    
        
    