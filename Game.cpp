#include "Game.hpp"
#include <unistd.h>
#include "gl_errors.hpp" //helper for dumpping OpenGL error messages
#include "read_chunk.hpp" //helper for reading a vector of structures from a file
#include "data_path.hpp" //helper to get paths relative to executable

#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <map>
#include <cstddef>
#include <random>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */


int boardState[14][11];
int playerPos[2] = {-1,-1};
int goalPos[2] = {-1,-1};
bool isMoving = false;
int direction[2] = {0,0}; 
int points = 0;
float timePassed = 0;

//helper defined later; throws if shader compilation fails:
static GLuint compile_shader(GLenum type, std::string const &source);


void resetGame(){
   for (int x=0; x<14;x++){
         for (int y=0; y<14; y++){  
            int rnum = arc4random_uniform(15);
            if (rnum >= 4) {
               rnum = 0;
            }
            boardState[x][y] = rnum;
            // 0 is plain
            // 1 is wall
            //2 is goop
            //3 is checkpoint

         }

      }

      playerPos[0] = arc4random_uniform(14);
      playerPos[1] =  arc4random_uniform(11);
      goalPos[0] = arc4random_uniform(14);
      goalPos[1] =  arc4random_uniform(11);
      while ((playerPos[0] == goalPos[0]) && (playerPos[1] == goalPos[1]) ){
         goalPos[0]  = arc4random_uniform(14);
         goalPos[1] =  arc4random_uniform(11);
      }
      boardState[playerPos[0]][playerPos[1]] = 0;
      boardState[goalPos[0]][goalPos[1]]=7;
      points = 0;

      isMoving=false;

}


Game::Game() {
   { //create an opengl program to perform sun/sky (well, directional+hemispherical) lighting:
      GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER,
            "#version 330\n"
            "uniform mat4 object_to_clip;\n"
            "uniform mat4x3 object_to_light;\n"
            "uniform mat3 normal_to_light;\n"
            "layout(location=0) in vec4 Position;\n" //note: layout keyword used to make sure that the location-0 attribute is always bound to something
            "in vec3 Normal;\n"
            "in vec4 Color;\n"
            "out vec3 position;\n"
            "out vec3 normal;\n"
            "out vec4 color;\n"
            "void main() {\n"
            "	gl_Position = object_to_clip * Position;\n"
            "	position = object_to_light * Position;\n"
            "	normal = normal_to_light * Normal;\n"
            "	color = Color;\n"
            "}\n"
            );

      GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER,
            "#version 330\n"
            "uniform vec3 sun_direction;\n"
            "uniform vec3 sun_color;\n"
            "uniform vec3 sky_direction;\n"
            "uniform vec3 sky_color;\n"
            "in vec3 position;\n"
            "in vec3 normal;\n"
            "in vec4 color;\n"
            "out vec4 fragColor;\n"
            "void main() {\n"
            "	vec3 total_light = vec3(0.0, 0.0, 0.0);\n"
            "	vec3 n = normalize(normal);\n"
            "	{ //sky (hemisphere) light:\n"
            "		vec3 l = sky_direction;\n"
            "		float nl = 0.5 + 0.5 * dot(n,l);\n"
            "		total_light += nl * sky_color;\n"
            "	}\n"
            "	{ //sun (directional) light:\n"
            "		vec3 l = sun_direction;\n"
            "		float nl = max(0.0, dot(n,l));\n"
            "		total_light += nl * sun_color;\n"
            "	}\n"
            "	fragColor = vec4(color.rgb * total_light, color.a);\n"
            "}\n"
            );

      simple_shading.program = glCreateProgram();
      glAttachShader(simple_shading.program, vertex_shader);
      glAttachShader(simple_shading.program, fragment_shader);
      //shaders are reference counted so this makes sure they are freed after program is deleted:
      glDeleteShader(vertex_shader);
      glDeleteShader(fragment_shader);

      //link the shader program and throw errors if linking fails:
      glLinkProgram(simple_shading.program);
      GLint link_status = GL_FALSE;
      glGetProgramiv(simple_shading.program, GL_LINK_STATUS, &link_status);
      if (link_status != GL_TRUE) {
         std::cerr << "Failed to link shader program." << std::endl;
         GLint info_log_length = 0;
         glGetProgramiv(simple_shading.program, GL_INFO_LOG_LENGTH, &info_log_length);
         std::vector< GLchar > info_log(info_log_length, 0);
         GLsizei length = 0;
         glGetProgramInfoLog(simple_shading.program, GLsizei(info_log.size()), &length, &info_log[0]);
         std::cerr << "Info log: " << std::string(info_log.begin(), info_log.begin() + length);
         throw std::runtime_error("failed to link program");
      }
   }

   { //read back uniform and attribute locations from the shader program:
      simple_shading.object_to_clip_mat4 = glGetUniformLocation(simple_shading.program, "object_to_clip");
      simple_shading.object_to_light_mat4x3 = glGetUniformLocation(simple_shading.program, "object_to_light");
      simple_shading.normal_to_light_mat3 = glGetUniformLocation(simple_shading.program, "normal_to_light");

      simple_shading.sun_direction_vec3 = glGetUniformLocation(simple_shading.program, "sun_direction");
      simple_shading.sun_color_vec3 = glGetUniformLocation(simple_shading.program, "sun_color");
      simple_shading.sky_direction_vec3 = glGetUniformLocation(simple_shading.program, "sky_direction");
      simple_shading.sky_color_vec3 = glGetUniformLocation(simple_shading.program, "sky_color");

      simple_shading.Position_vec4 = glGetAttribLocation(simple_shading.program, "Position");
      simple_shading.Normal_vec3 = glGetAttribLocation(simple_shading.program, "Normal");
      simple_shading.Color_vec4 = glGetAttribLocation(simple_shading.program, "Color");
   }

   struct Vertex {
      glm::vec3 Position;
      glm::vec3 Normal;
      glm::u8vec4 Color;
   };
   static_assert(sizeof(Vertex) == 28, "Vertex should be packed.");

   { //load mesh data from a binary blob:
      std::ifstream blob(data_path("meshes.blob"), std::ios::binary);
      //The blob will be made up of three chunks:
      // the first chunk will be vertex data (interleaved position/normal/color)
      // the second chunk will be characters
      // the third chunk will be an index, mapping a name (range of characters) to a mesh (range of vertex data)

      //read vertex data:
      std::vector< Vertex > vertices;
      read_chunk(blob, "dat0", &vertices);

      //read character data (for names):
      std::vector< char > names;
      read_chunk(blob, "str0", &names);

      //read index:
      struct IndexEntry {
         uint32_t name_begin;
         uint32_t name_end;
         uint32_t vertex_begin;
         uint32_t vertex_end;
      };
      static_assert(sizeof(IndexEntry) == 16, "IndexEntry should be packed.");

      std::vector< IndexEntry > index_entries;
      read_chunk(blob, "idx0", &index_entries);

      if (blob.peek() != EOF) {
         std::cerr << "WARNING: trailing data in meshes file." << std::endl;
      }

      //upload vertex data to the graphics card:
      glGenBuffers(1, &meshes_vbo);
      glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      //create map to store index entries:
      std::map< std::string, Mesh > index;
      for (IndexEntry const &e : index_entries) {
         if (e.name_begin > e.name_end || e.name_end > names.size()) {
            throw std::runtime_error("invalid name indices in index.");
         }
         if (e.vertex_begin > e.vertex_end || e.vertex_end > vertices.size()) {
            throw std::runtime_error("invalid vertex indices in index.");
         }
         Mesh mesh;
         mesh.first = e.vertex_begin;
         mesh.count = e.vertex_end - e.vertex_begin;
         auto ret = index.insert(std::make_pair(
                  std::string(names.begin() + e.name_begin, names.begin() + e.name_end),
                  mesh));
         if (!ret.second) {
            throw std::runtime_error("duplicate name in index.");
         }
      }

      //look up into index map to extract meshes:
      auto lookup = [&index](std::string const &name) -> Mesh {
         auto f = index.find(name);
         if (f == index.end()) {
            throw std::runtime_error("Mesh named '" + name + "' does not appear in index.");
         }
         return f->second;
      };
      checkpoint_mesh = lookup("Checkpoint");
      floor_mesh = lookup("Floor");
      wall_mesh = lookup("Wall");
      goal_mesh = lookup("Goal");
      player_mesh = lookup("Player");
      goop_mesh = lookup("Goop");
   }

   { //create vertex array object to hold the map from the mesh vertex buffer to shader program attributes:
      glGenVertexArrays(1, &meshes_for_simple_shading_vao);
      glBindVertexArray(meshes_for_simple_shading_vao);
      glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
      //note that I'm specifying a 3-vector for a 4-vector attribute here, and this is okay to do:
      glVertexAttribPointer(simple_shading.Position_vec4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Position));
      glEnableVertexAttribArray(simple_shading.Position_vec4);
      if (simple_shading.Normal_vec3 != -1U) {
         glVertexAttribPointer(simple_shading.Normal_vec3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Normal));
         glEnableVertexAttribArray(simple_shading.Normal_vec3);
      }
      if (simple_shading.Color_vec4 != -1U) {
         glVertexAttribPointer(simple_shading.Color_vec4, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Color));
         glEnableVertexAttribArray(simple_shading.Color_vec4);
      }
      glBindBuffer(GL_ARRAY_BUFFER, 0);
   }

   GL_ERRORS();

   //----------------
   //set up game board with meshes and rolls:
   board_meshes.reserve(board_size.x * board_size.y);
   board_rotations.reserve(board_size.x * board_size.y);
   std::mt19937 mt(0xbead1234);

   auto goalReachable = [&](){
      if (playerPos[0]==-1){return false;}
      else{
         return true;} 
   };        

   while(goalReachable() == false){ 
      resetGame();
      }



   std::vector< Mesh const * > meshes{ &player_mesh, &checkpoint_mesh, &goal_mesh };

   for (uint32_t i = 0; i < board_size.x * board_size.y; ++i) {
      board_meshes.emplace_back(meshes[mt()%meshes.size()]);
      board_rotations.emplace_back(glm::quat());
   }

}

Game::~Game() {
   glDeleteVertexArrays(1, &meshes_for_simple_shading_vao);
   meshes_for_simple_shading_vao = -1U;

   glDeleteBuffers(1, &meshes_vbo);
   meshes_vbo = -1U;

   glDeleteProgram(simple_shading.program);
   simple_shading.program = -1U;

   GL_ERRORS();
}

bool Game::handle_event(SDL_Event const &evt, glm::uvec2 window_size) {
  //ignore any keys that are the result of automatic key repeat:
   if (evt.type == SDL_KEYDOWN && evt.key.repeat) {
      return false;
   }
   //handle tracking the state of WSAD for roll control:
   if (evt.type == SDL_KEYDOWN || evt.type == SDL_KEYUP) {
      if (evt.key.keysym.scancode == SDL_SCANCODE_W) {
         controls.roll_up = (evt.type == SDL_KEYDOWN);
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_S) {
         controls.roll_down = (evt.type == SDL_KEYDOWN);
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_A) {
         controls.roll_left = (evt.type == SDL_KEYDOWN);
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_D) {
         controls.roll_right = (evt.type == SDL_KEYDOWN);
         return true;
      }
      else if (evt.key.keysym.scancode == SDL_SCANCODE_R) {
         controls.reset = (evt.type == SDL_KEYDOWN);
         return true;
      }

   }
   //move cursor on L/R/U/D press:
   if (evt.type == SDL_KEYUP && evt.key.repeat == 0) {
      if (evt.key.keysym.scancode == SDL_SCANCODE_LEFT) {
         if (cursor.x > 0) {
            cursor.x -= 1;
         }
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_RIGHT) {
         if (cursor.x + 1 < board_size.x) {
            cursor.x += 1;
         }
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_UP) {
         if (cursor.y + 1 < board_size.y) {
            cursor.y += 1;
         }
         return true;
      } else if (evt.key.keysym.scancode == SDL_SCANCODE_DOWN) {
         if (cursor.y > 0) {
            cursor.y -= 1;
         }
         return true;
      }
   }
   return false;
}

void Game::update(float elapsed) {

  
   if (isMoving){ //process collisions and add up points
 if (timePassed+elapsed<0.10){
   timePassed = timePassed + elapsed;
   return;   
}
else{
   timePassed = timePassed+elapsed-0.10;
}
      
      int nextX = playerPos[0] + direction[0];
      int nextY = playerPos[1] + direction[1];
      int nextBlockType = boardState[nextX][nextY];
      if ((nextX<0 || nextX>13) || (nextY<0 || nextY>10)){ nextBlockType = 1;}



      switch(nextBlockType){
         case 1:   //wall
            isMoving = false;
            break;
         case 2:   //goop
            isMoving = false;
            //boardState[playerPos[0]][playerPos[1]] = 0;
            playerPos[0] = nextX;
               playerPos[1] = nextY;
               //boardState[nextX][nextY] = 6 ;
               break;
         case 3:   //checkpoint
            //boardState[playerPos[0]][playerPos[1]] = 0;
               playerPos[0] = nextX;
               playerPos[1] = nextY;

               boardState[nextX][nextY] = 0; 
               points = points+1;
               break;
         case 7:   //goal
            isMoving = false;
              // boardState[playerPos[0]][playerPos[1]] = 0;
               playerPos[0] = nextX;
               playerPos[1] = nextY;


               //boardState[nextX][nextY] = ; 
               break;
         case 0:   //nothing
            //boardState[playerPos[0]][playerPos[1]] = 0;
               playerPos[0] = nextX;
               playerPos[1] = nextY;

               //boardState[nextX][nextY] = 6;            
               break;}  


   }
   else{

      //if the roll keys are pressed, rotate everything on the same row or column as the cursor:
      if (controls.roll_left) {
         isMoving = true;
         direction[0] = -1;
         direction[1] = 0; 
         timePassed=0;
         return;
         //	dr = glm::angleAxis(amt, glm::vec3(0.0f, 1.0f, 0.0f)) * dr;
      }
      if (controls.roll_right) {
         isMoving = true;
         direction[0] = 1;
         direction[1] = 0;
      timePassed=0;
         return;
        
     //	dr = glm::angleAxis(-amt, glm::vec3(0.0f, 1.0f, 0.0f)) * dr;
      }
      if (controls.roll_up) {
         isMoving = true;
         direction[0] = 0;
         direction[1] = 1;
      timePassed=0;
         return;
          
   //	dr = glm::angleAxis(amt, glm::vec3(1.0f, 0.0f, 0.0f)) * dr;
      }
      if (controls.roll_down) {
         isMoving = true;
         direction[0] = 0;
         direction[1] = -1;
      timePassed=0;
         return;
     
        //	dr = glm::angleAxis(-amt, glm::vec3(1.0f, 0.0f, 0.0f)) * dr;
      }
      if (controls.reset){
         resetGame();
         return;

      }
   
   }
}

void Game::draw(glm::uvec2 drawable_size) {
   //Set up a transformation matrix to fit the board in the window:
   glm::mat4 world_to_clip;
   {
      float aspect = float(drawable_size.x) / float(drawable_size.y);

      //want scale such that board * scale fits in [-aspect,aspect]x[-1.0,1.0] screen box:
      float scale = glm::min(
            2.0f * aspect / float(board_size.x),
            2.0f / float(board_size.y)
            );

      //center of board will be placed at center of screen:
      glm::vec2 center = 0.5f * glm::vec2(board_size);

      //NOTE: glm matrices are specified in column-major order
      world_to_clip = glm::mat4(
            scale / aspect, 0.0f, 0.0f, 0.0f,
            0.0f, scale, 0.0f, 0.0f,
            0.0f, 0.0f,-1.0f, 0.0f,
            -(scale / aspect) * center.x, -scale * center.y, 0.0f, 1.0f
            );
   }

   //set up graphics pipeline to use data from the meshes and the simple shading program:
   glBindVertexArray(meshes_for_simple_shading_vao);
   glUseProgram(simple_shading.program);

   glUniform3fv(simple_shading.sun_color_vec3, 1, glm::value_ptr(glm::vec3(0.81f, 0.81f, 0.76f)));
   glUniform3fv(simple_shading.sun_direction_vec3, 1, glm::value_ptr(glm::normalize(glm::vec3(-0.2f, 0.2f, 1.0f))));
   glUniform3fv(simple_shading.sky_color_vec3, 1, glm::value_ptr(glm::vec3(0.2f, 0.2f, 0.3f)));
   glUniform3fv(simple_shading.sky_direction_vec3, 1, glm::value_ptr(glm::vec3(0.0f, 1.0f, 0.0f)));

   //helper function to draw a given mesh with a given transformation:
   auto draw_mesh = [&](Mesh const &mesh, glm::mat4 const &object_to_world) {
      //set up the matrix uniforms:
      if (simple_shading.object_to_clip_mat4 != -1U) {
         glm::mat4 object_to_clip = world_to_clip * object_to_world;
         glUniformMatrix4fv(simple_shading.object_to_clip_mat4, 1, GL_FALSE, glm::value_ptr(object_to_clip));
      }
      if (simple_shading.object_to_light_mat4x3 != -1U) {
         glUniformMatrix4x3fv(simple_shading.object_to_light_mat4x3, 1, GL_FALSE, glm::value_ptr(object_to_world));
      }
      if (simple_shading.normal_to_light_mat3 != -1U) {
         //NOTE: if there isn't any non-uniform scaling in the object_to_world matrix, then the inverse transpose is the matrix itself, and computing it wastes some CPU time:
         glm::mat3 normal_to_world = glm::inverse(glm::transpose(glm::mat3(object_to_world)));
         glUniformMatrix3fv(simple_shading.normal_to_light_mat3, 1, GL_FALSE, glm::value_ptr(normal_to_world));
      }

      //draw the mesh:
      glDrawArrays(GL_TRIANGLES, mesh.first, mesh.count);
   };

   auto  draw_board = [&](){
      for (int x=0; x<14;x++){
         for (int y=0; y<11; y++){
            switch(boardState[x][y]) {
               case 0 : /*draw_mesh(floor_mesh, glm::mat4(
                          0.4f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.4f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          float(x)+1.0, float(y)+1.0, 0.5f, 3.0f
                          ));*/
                  break;
               case 1 : draw_mesh(wall_mesh, glm::mat4(
                                 0.4f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.4f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f,
                                 float(x)+1.0, float(y)+1.0, 0.5f, 3.0f
                                 ));
                        break;

               case 2: draw_mesh(goop_mesh, glm::mat4(
                                0.4f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.4f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                float(x)+1.0, float(y)+1.0, 0.5f, 3.0f
                                ));
                       break;
               case 3: draw_mesh(checkpoint_mesh, glm::mat4(
                                0.4f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.4f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                float(x)+1.0, float(y)+1.0, 0.5f, 3.0f
                                ));
                       break;
              case 7: draw_mesh(goal_mesh, glm::mat4(
                                0.4f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.4f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                float(x)+1.0, float(y)+1.0, 0.5f, 3.0f
                                ));
                       break;

            } 


         }

      }

      draw_mesh(player_mesh, glm::mat4(
                                0.4f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.4f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                playerPos[0]+1.0, playerPos[1]+1.0, 0.75f, 3.0f
                                ));
   
      //draw game points
      for (int i=0; i<points; i++){
 draw_mesh(checkpoint_mesh, glm::mat4(
                                0.05f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.05f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                15.0, 11.0-0.25*i, 0.75f, 3.0f
                                ));
   


      }
              
   };
   draw_board();


   glUseProgram(0);

   GL_ERRORS();
}

//create and return an OpenGL vertex shader from source:
static GLuint compile_shader(GLenum type, std::string const &source) {
   GLuint shader = glCreateShader(type);
   GLchar const *str = source.c_str();
   GLint length = GLint(source.size());
   glShaderSource(shader, 1, &str, &length);
   glCompileShader(shader);
   GLint compile_status = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
   if (compile_status != GL_TRUE) {
      std::cerr << "Failed to compile shader." << std::endl;
      GLint info_log_length = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
      std::vector< GLchar > info_log(info_log_length, 0);
      GLsizei length = 0;
      glGetShaderInfoLog(shader, GLsizei(info_log.size()), &length, &info_log[0]);
      std::cerr << "Info log: " << std::string(info_log.begin(), info_log.begin() + length);
      glDeleteShader(shader);
      throw std::runtime_error("Failed to compile shader.");
   }
   return shader;
}
