#include"./ParticleSystem.h"

// 主函数：窗口初始化与主循环
int main() {
    // 创建SFML窗口
    sf::RenderWindow window(
    sf::VideoMode(sf::Vector2u(1600, 1000)),  // 显式传递 Vector2u
    "GPU粒子系统（使用Particle类）",
    sf::Style::Close
);
    window.setFramerateLimit(60);  // 限制帧率为60FPS
    sf::Clock clock;               // 用于计算帧间隔时间

    // 初始化粒子系统
    ParticleSystem particleSystem(window.getSize().x, window.getSize().y);

    // 主循环
    while (window.isOpen()) {
        // 计算帧间隔时间（秒）
        float deltaTime = clock.restart().asSeconds();
        // 事件处理（关闭窗口）
        while (const std::optional<sf::Event> event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>()) 
            {
                window.close();
            } 
            else if (event->is<sf::Event::KeyPressed>()) 
            {
                if (const auto* keyEvent = event->getIf<sf::Event::KeyPressed>()) 
                {
                    if (keyEvent->code == sf::Keyboard::Key::Delete) 
                    {
                        window.close();
                    }
                }
            } 
            else if (const auto* resizeEvent = event->getIf<sf::Event::Resized>()) 
            {
                sf::Vector2u newSize = resizeEvent->size;
                // 处理窗口大小变化...
                
            }
        }
            // 清空窗口（黑色背景）
            window.clear(sf::Color::Black);

            // 更新并渲染粒子
            particleSystem.updateAndRender(window, deltaTime);

            // 显示当前帧
            window.display();
        }

    return 0;
    }

    