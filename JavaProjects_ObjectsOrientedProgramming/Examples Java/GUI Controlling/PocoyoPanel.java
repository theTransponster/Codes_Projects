import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.ImageIcon;
import javax.swing.JPanel;

public class PocoyoPanel extends JPanel implements Runnable{

	private int xV,
				yV,
				xT,
				yT;
	private Image fondo;
	private String nombre="Pocoyo";
	private boolean movernave;
	private Color color;
	
	public PocoyoPanel(){
		super();
		this.setPreferredSize(new Dimension (800,600));
		this.movernave=false;
		Thread hilo=new Thread(this);
		hilo.start();
		this.addMouseListener(new MouseAdapter(){
			public void mouseClicked(MouseEvent e){
				System.out.println("Se dio click en: "+e.getX()+", "+e.getY());
				movernave=true;
			}
		});
		this.addMouseMotionListener(new MouseAdapter(){
			public void mouseDragged(MouseEvent arg0) {//aqui se crea una clase anonima
				System.out.println("Arrastrando el mouse en: "+arg0.getX()+","+arg0.getY());
				xT=arg0.getX();
				yT=arg0.getY();
				repaint();
			}
		});
		this.fondo= new ImageIcon(getClass().getResource("pocoyo2.jpg")).getImage();
		this.color=Color.RED;
		this.xT=110;
		this.yT=570;
	}
	public void paintComponent(Graphics g){
		super.paintComponent(g);
		g.drawImage(this.fondo,0,0,this.getWidth(),this.getHeight(),this);
		g.setColor(Color.ORANGE);
		g.fillOval(50+this.xV, 500-this.yV, 200, 100);
		g.setColor(Color.BLACK);
		g.drawLine(100+this.xV, 507-this.yV, 100+this.xV, 407-this.yV);
		g.setColor(this.color);
		g.fillOval(50+this.xV, 307-this.yV, 100, 100);
		g.setColor(Color.CYAN);
		g.fillArc(50+this.xV, 500-this.yV, 200, 100, 0, 100);
		g.setColor(Color.BLACK);
		g.drawString("Vamos "+this.nombre+" !!", this.xT, this.yT);
	}
	public void run(){
		try{
			while(this.xV<550){
				if(this.movernave){
					this.xV+=2;
					this.yV++;
					this.repaint();
				}
				Thread.sleep(15);
			}	
		}catch(InterruptedException e){
			System.out.println("No se pudo ejecutar");
			}
		}
	public void getNombre(String s){
		this.nombre=s;
		this.repaint();
	}
	public void getColor(Color color){
		this.color=color;
		this.repaint();
	}

}
