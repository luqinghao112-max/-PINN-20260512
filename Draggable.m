% --- UI 轻量级自由拖拽引擎 ---
function Draggable(h_obj)
    set(h_obj, 'ButtonDownFcn', @start_drag);
    
    function start_drag(src, ~)
        fig = ancestor(src, 'figure');
        fig.Units = 'pixels';
        src.Units = 'pixels';
        uistack(src, 'top'); 
        src.UserData = struct('StartPos', fig.CurrentPoint, 'StartBox', src.Position);
        set(fig, 'WindowButtonMotionFcn', @(f,e) drag_move(src, f));
        set(fig, 'WindowButtonUpFcn', @(f,e) drag_stop(src, f));
    end
    function drag_move(src, fig)
        cp = fig.CurrentPoint;
        ud = src.UserData;
        dx = cp(1) - ud.StartPos(1);
        dy = cp(2) - ud.StartPos(2);
        src.Position = [ud.StartBox(1)+dx, ud.StartBox(2)+dy, ud.StartBox(3), ud.StartBox(4)];
    end
    function drag_stop(src, fig)
        set(fig, 'WindowButtonMotionFcn', '');
        set(fig, 'WindowButtonUpFcn', '');
        src.Units = 'normalized'; 
        fig.Units = 'normalized';
    end
end