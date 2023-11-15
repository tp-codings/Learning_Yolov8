
def isFullHouse(ranks):
    ranks_count = {}
    
    for rank in ranks:
        ranks_count[rank] = ranks_count.get(rank, 0) + 1

    return sorted(ranks_count.values()) == [2, 3] or sorted(ranks_count.values()) == [3, 2]

def getMaxOccurences(ranks):
    occurrences = {}
    
    for rank in ranks:
        occurrences[rank] = occurrences.get(rank, 0) + 1

    max_occurrence = max(occurrences.values(), default=0)

    elements_with_max_occurrence = [element for element, count in occurrences.items() if count == max_occurrence]

    return max_occurrence, len(elements_with_max_occurrence) == 2

def findPokerHand(hand):

    ranks = []
    suits = []

    for card in hand:

        if len(card) == 3:
            rank = card[0:2]
            suit = card[2]
        else:
            rank = card[0]
            suit = card[1]

        if rank == 'J':
            rank = 11
        elif rank == 'Q':
            rank = 12
        elif rank == 'K':
            rank = 13
        elif rank == 'A':
            rank = 14

        ranks.append(rank)
        suits.append(suit)
    
    ranks = list(map(int, ranks))

    maxOcc, twoPair = getMaxOccurences(ranks)

    result = "High card"
    if maxOcc == 2:
        result = "Pair"

    if twoPair:
        result = "Two pair"

    if maxOcc == 3:
        result = "Three of a Kind"

    if sorted(ranks) == list(range(min(ranks), max(ranks)+1)): #Ass == 1 not detectable here
        result = "Straight"

    if suits.count(suits[0]) == 5:
        result = "Flush"
        
    if sorted(ranks) == list(range(min(ranks), max(ranks)+1)) and result == "Flush":
        result = "Straight Flush"
    if 14 in ranks and result == "Straight Flush":
        result = "Royal Flush"

    if(isFullHouse(ranks)):
        result = "Full House"

    if maxOcc == 4:
        result = "Four of a Kind"

    return result

if __name__ == "__main__":
    findPokerHand(["AH", "KH", "QH", "JH", "10H"])  #RF
    findPokerHand(["QC", "JC", "10C", "9C", "8C"])  #SF
    findPokerHand(["5D", "5C", "5S", "5H", "8C"])   #4K
    findPokerHand(["5D", "5C", "5S", "2H", "2C"])   #FH
    findPokerHand(["5D", "7D", "3D", "JD", "4D"])   #F
    findPokerHand(["5D", "7D", "4H", "8D", "6D"])   #S
    findPokerHand(["5D", "5H", "4H", "5S", "6D"])   #3K
    findPokerHand(["5D", "5H", "4H", "4S", "6D"])   #2P
    findPokerHand(["5D", "5H", "4H", "8S", "6D"])   #2K
    findPokerHand(["5D", "9H", "4H", "8S", "6D"])   #HK