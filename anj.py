class Room {
    std::string description;
    std::vector<Entity*> entities;
    std::map<std::string, Room*> exits;

public:
    Room(const std::string& desc) : description(desc) {}
    void addEntity(Entity* e) { entities.push_back(e); }
    void setExit(const std::string& direction, Room* room) { exits[direction] = room; }
    // Additional methods for interaction and movement...
};
class Dungeon {
    Room* entrance;

public:
    void setEntrance(Room* entry) { entrance = entry; }
    // Methods for constructing the dungeon, adding rooms, etc...
};

void exploreDungeon(Dungeon& dungeon) {
    Room* currentRoom = dungeon.getEntrance();
    std::string command;

    while (currentRoom) {
        std::cout << "You are in: " << currentRoom->getDescription() << std::endl;
        std::cin >> command;
        // Logic to move to the next room based on command...
    }
}


