import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JOptionPane;
import javax.swing.JPanel;

public class FiguraPanel extends JPanel {
	
	
	private Color color;
	private String fig;
	private Figura f;
	private int x,
				y,
				x2,
				y2;
	private boolean mouse;
	
	public FiguraPanel(){
		super();
		this.setPreferredSize(new Dimension(800,600));
		this.color=Color.black;
		this.fig="";
		
		this.y=0;
		this.x=0;
		this.mouse=false;
		
		
	}
	public void paintComponent(Graphics g){
		super.paintComponent(g);
		
		g.setColor(Color.black);
		
		g.setColor(this.color);
		if(this.mouse==false){
			g.drawString(this.fig, this.x,this.y-10);
			if(this.f instanceof Cuadrado){
				g.drawRect(this.x, this.y, (int)((Cuadrado) this.f).largo, (int)((Cuadrado) this.f).largo);
			}else if(this.f instanceof Caja){
				g.fillRect(this.x,this.y,(int)((Caja) this.f).ancho,(int)((Caja) this.f).largo);
				g.drawRect(this.x, this.y, (int)(((Caja) this.f).ancho+(((Caja) this.f).alto)), (int)(((Caja) this.f).largo+(((Caja) this.f).alto)));
			}else if(this.f instanceof Rectangulo){
				g.fillRect(this.x, this.y, (int)((Rectangulo) this.f).ancho, (int)((Rectangulo) this.f).largo);
			}
		}else{
			g.drawRect(this.x, this.y, this.x2-this.x, this.y2-this.y);
		}
	}
	
	public void setColor(Color color){
		this.color=color;
		this.repaint();
	}
	public void setString(String figura){
		this.fig=figura;
		this.repaint();
	}
	public void setFigura(Figura fig){
		this.f=fig;
		this.repaint();
	}
	public void setCoordenadas(int x, int y){
		this.x=x;
		this.y=y;
		this.repaint();
	}
	public void setModo(Boolean m){
		this.mouse=m;
		this.repaint();
	}
	public void setFinales(int x2, int y2){
		this.x2=x2;
		this.y2=y2;
		this.repaint();
	}
	
}
