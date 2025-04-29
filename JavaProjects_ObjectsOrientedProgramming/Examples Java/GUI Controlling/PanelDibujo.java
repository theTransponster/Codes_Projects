import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.ImageIcon;
import javax.swing.JPanel;
//panel es como una ventana pero sin marco

public class PanelDibujo extends JPanel implements Runnable{
	private int xV,
				yV,
				xLabel,
				yLabel;
	private Color colorglobo=Color.RED; 
	private Color colornave=Color.YELLOW;
	private ImageIcon fondo;
	private boolean moverVamoosh;
	private String nombre="Pocoyo";
	public PanelDibujo(){
		super();
		this.setPreferredSize(new Dimension(800,600));
		this.addMouseListener(new MouseListener(){

			@Override
			public void mouseClicked(MouseEvent arg0) {
				// TODO Auto-generated method stub
				moverVamoosh=true;
			}

			@Override
			public void mouseEntered(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mouseExited(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mousePressed(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mouseReleased(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}
			
		});
		
		/*this.addMouseMotionListener(new MouseMotionListener() {
			
			@Override
			public void mouseMoved(MouseEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void mouseDragged(MouseEvent e) {
				//System.out.println("Hola en "+e.getX()+","+e.getY());
				xLabel=e.getX();
				yLabel=e.getY();
				repaint();	
			}
		});*/
		this.moverVamoosh=false;
		
		
		this.xV=0;
		this.yV=0;
		Thread hilo=new Thread(this);
		hilo.start();
	}
	
	public void paintComponent(Graphics g){ //le pasa como parametro el contexto grafico
		super.paintComponent(g);
		this.fondo=new ImageIcon(getClass().getResource("maxresdefault.jpg"));
		g.drawImage(this.fondo.getImage(),0,0,this.getWidth(),this.getHeight(),null);
		g.setColor(this.colornave);
		g.fillOval(50+this.xV, 500-this.yV, 200, 100);
		g.setColor(Color.BLACK);
		g.drawLine(100+this.xV, 507-this.yV, 100+this.xV, 407-this.yV);
		g.setColor(this.colorglobo);
		g.fillOval(50+this.xV, 307-this.yV, 100, 100);
		g.setColor(Color.BLACK);
		g.drawString("Hola "+this.nombre, 100+this.xV, 550-this.yV);
		
	}
		public void run() {
			try {
				while(xV<550) {
					if(moverVamoosh) {
						xV+=2;
						yV++;
						repaint();
					}
					Thread.sleep(20);
				}
			}
			catch(InterruptedException e) {
				System.out.println("No se pudo ejecutar");
			}
		}
	public void setiVY(int valor){
		this.yV=valor;
		this.repaint();
	}

	public void setColor(Color colorglobo){
		this.colorglobo=colorglobo;
		this.repaint();
	}
	public Color getColor(){
		return this.colorglobo;
	}
	public void setColornave(Color colornave){
		this.colornave=colornave;
		this.repaint();
	}
	public Color getColornave(){
		return this.colornave;
	}
	public void nombre(String nombre){
		this.nombre=nombre;
		this.repaint();
	}

}
